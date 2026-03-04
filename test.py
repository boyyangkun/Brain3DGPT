import argparse
import os
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_auc_score
from train_utils import set_seed
import json
import numpy as np
import torch
from torch.cuda.amp import autocast as autocast
from train_utils import inference, cal_metrics, dice_metric, post_trans
from dataset.multimodal_dataset import format_input
from dataset.transforms import custom_transform
from model.ap.openllama import OpenLLAMAPEFTModel
from loss.dice import EDiceLoss
from monai.data import DataLoader, Dataset, decollate_batch
from monai.data.utils import pad_list_data_collate
from evaluatemain.metrics.bleu.bleu import Bleu
import nibabel as nib
import time
import random
import csv
import evaluate
import nltk
nltk.data.path.append("/home/aiotlab/nltk_data")

parser = argparse.ArgumentParser(description='BrainMVP downstream segmentation testing')

parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--patch_shape', default=96, type=int, help='input shape, default to 96')
parser.add_argument('--in_channels', default=4, type=int, help="channels of model inputs")
parser.add_argument('--out_classes', default=3, type=int, help="channels of model outputs")
parser.add_argument('--checkpoint', default='./debug/ap/model_best_42.pth.tar', type=str, help='path of checkpoint to load from')

parser.add_argument('--workers', default=2, type=int, help='number of workers, default to 4')                    
parser.add_argument('--devices', default='0', type=str, help='cuda visible devices, default to 0')
parser.add_argument("--local_rank", type=int, default=0, help="local rank")
parser.add_argument('--random_seed', type=int,default=42, help='random seed')
parser.add_argument('--dataset', type=str, default='upenngbm', choices=['brats18', 'brats2021', 'brats23-ped', 'brats23-met',
                                                                        'isles22','mrbrains13', 'vsseg', 'upenngbm'])
parser.add_argument('--data_root', default='', type=str, help='root path to images')
parser.add_argument('--json_file', default='dataset_upenngbm_ds.json', type=str, help='json file name')

parser.add_argument('--multi_scale', default=False, type=bool, help='whether to apply multi_scale')
parser.add_argument('--Qwen_ckpt_path', default='', type=str, help='path to Qwen checkpoint')
parser.add_argument('--max_tgt_len', default=1024, type=int,help='the maximum sequence length')
parser.add_argument('--pretrained', default='', type=str, help="Pretrained weight path")
parser.add_argument('--lora_r', default=16, type=int)
parser.add_argument('--lora_alpha', default=64, type=int)
parser.add_argument('--lora_dropout', default=0.05, type=float)
parser.add_argument('--delta_ckpt_path', default='./7b_v0/', type=str, help='path to pandagpt checkpoint')
parser.add_argument('--csv_name', default='detection_result', type=str, help='result')

# txt
# bleu = evaluate.load("./evaluate-main/metrics/bleu/bleu.py")
# print(1)
bleu = Bleu()
bertscore = evaluate.load("./evaluatemain/metrics/bertscore/bertscore.py")
meteor = evaluate.load("./evaluatemain/metrics/meteor/meteor.py")
rouge = evaluate.load("./evaluatemain/metrics/rouge/rouge.py")


def postprocess_text(preds, labels):
    # preds: List[List[dict]]，只取 gpt 的 value
    processed_preds = []
    for sample in preds:
        # sample 是一个 list of dict
        gpt_texts = [item['value'] for item in sample if item['from'] == 'gpt']
        # 拼接多条 gpt 文本（如果有多条）
        text = " ".join(gpt_texts).strip()
        processed_preds.append(text)

    # labels 处理成二维列表，并去掉首尾空格
    processed_labels = [[labels.strip()]]

    return processed_preds, processed_labels

def run_inference(data_loader, metric, top_p, temperature, max_length, name):
    metrics = []
    acc = 0
    # --- 图像级别 AUC 使用 ---
    slice_probs, slice_labels = [], []
    all_probs_flat, all_labels_flat = [], []
    alltime = 0
    bleu_score_all, rouge_score_all, bert_score_all, meteor_score_all = [],[],[],[]
    n = 0
    for val_data in tqdm(data_loader):
        n += 1
        model.eval()
        with torch.no_grad():
            val_inputs = val_data["image"].cuda()
            val_labels = val_data["label"].cuda()
            val_texts = val_data["text"]
            decoded_labels = [d['value'][0] for d in val_texts if d['from'][0] == 'gpt']
            prompt_text = f'{propmt}'
            loop_start_time = time.time()
            try:
                response, val_outputs = model.generate({
                    'prompt': prompt_text,
                    'image_paths': val_inputs,
                    'top_p': top_p,
                    'temperature': temperature,
                    'max_tgt_len': max_length,
                })
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                print(f"Skipping batch due to error: {e}")
                torch.cuda.empty_cache()
                continue
            try:
                stage1_total_time = time.time() - loop_start_time
                alltime += stage1_total_time
                decoded_preds = [response.strip()]
                bleu_score = bleu.compute(predictions=decoded_preds, references=[decoded_labels], max_order=1)
                bleu_score_all.append(bleu_score['bleu'])

                rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels, rouge_types=['rouge1'])
                rouge_score_all.append(rouge_score['rouge1'])

                meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_labels)
                meteor_score_all.append(meteor_score['meteor'])

                bert_score = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
                bert_score_all.append(sum(bert_score['f1']) / len(bert_score['f1']))
                
                # val_preds = [post_trans(i) for i in decollate_batch(val_outputs)]
                # dice_metric(y_pred=val_preds, y=val_labels)
                # val_probs = torch.sigmoid(val_outputs)
                # prob_flat = val_probs.flatten().cpu().numpy()
                # label_flat = val_labels.flatten().cpu().numpy()
                # all_probs_flat.extend(prob_flat)
                # all_labels_flat.extend(label_flat)
            except Exception as e:
                print(f"警告: 样本计算错误 {e}")
                continue

        #     # 对每个 sample 统一尺寸
        #     for prob, label in zip(val_probs.cpu(), val_labels.cpu()):
        #         C, H, W, D = prob.shape
        #         for d in range(D):
        #             prob_slice = prob[..., d]  # shape: (C, H, W)
        #             label_slice = label[..., d]

        #             # 计算该切片的异常概率为预测值的最大值
        #             anomaly_score = prob_slice.max().item()
                    
        #             # 标注该切片是否有异常（label中只要有非零就是异常）
        #             anomaly_label = 1 if label_slice.sum() > 0.3 else 0
        #             # print(anomaly_label,anomaly_score)
        #             slice_probs.append(anomaly_score)
        #             slice_labels.append(anomaly_label)

        # metric_ = metric(val_outputs, val_labels)
        # metrics.extend(metric_)
    print(
                "bleu: {:.4f}".format(sum(bleu_score_all) / len(bleu_score_all)),
                "rouge1: {:.4f}".format(sum(rouge_score_all) / len(rouge_score_all)),
                "meteor: {:.4f}".format(sum(meteor_score_all) / len(meteor_score_all)),
                "bert_f1: {:.4f}".format(sum(bert_score_all) / len(bert_score_all)),
            )
    print(alltime)
    # all_probs_flat = np.array(all_probs_flat)
    # all_labels_flat = np.array(all_labels_flat)
    # pixel_auc = roc_auc_score(all_labels_flat, all_probs_flat)
    # slice_auc = roc_auc_score(slice_labels, slice_probs)
    # print('Dice metric: ', dice_metric.aggregate().mean().item(), 'slice_auc: ', slice_auc, 'piex_AUC: ', pixel_auc)
    # class_avg_metrics = cal_metrics(metrics, acc, mode='test')

    # dice_metric.reset()


def get_test_loader(args):
    data_root = Path(args.data_root)
    with open(os.path.join(data_root, args.json_file), 'r') as fr:
        data_list = json.load(fr)
    test_list = format_input(data_root, data_list["test"])
    print(f'Test length: {len(test_list)}')
    test_transform = custom_transform(patch_shape=args.patch_shape, mode='test')
    test_dataset = Dataset(data=test_list, transform=test_transform)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn = pad_list_data_collate, pin_memory=torch.cuda.is_available())
    return val_loader


input = "ls there any anomaly in this brainMRI?"
describles = "This is a brain MRI image intended for anomaly detection. The image should be free from any signs of pathology, without tumors, lesions, hemorrhages, or other abnormalities."
propmt = describles + ' ' + input
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.devices
args.rank = 0
set_seed(args)
print("Building model ...")
model = OpenLLAMAPEFTModel(args)

ckpt = torch.load(args.checkpoint, map_location=torch.device('cpu'))
model.load_state_dict(ckpt['state_dict'], strict=False)
model = model.cuda()

loss_func = EDiceLoss().cuda()
eval_metrics = loss_func.metric

test_loader = get_test_loader(args)
run_inference(test_loader, eval_metrics, 0.9, 1.0, 128, args.csv_name)




