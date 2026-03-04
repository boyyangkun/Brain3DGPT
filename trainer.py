import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import time
import numpy as np
import torch
from torch.nn import functional as F
from train_utils import AverageMeter, save_checkpoint, inference, post_trans, cal_metrics, cal_save
from monai.data import decollate_batch
from tqdm import tqdm
from torch.cuda.amp import autocast
import evaluate
import nltk
from evaluatemain.metrics.bleu.bleu import Bleu
from transformers import logging
import warnings
# 在代码开头添加
warnings.filterwarnings("ignore", message=".*resume_download.*")
logging.set_verbosity_error()
nltk.data.path.append("/home/aiotlab/nltk_data")
def train_epoch(epoch, model, loader, optimizer, scheduler, loss_func, scaler, args):
    # Setup
    train_total_loss = AverageMeter('Loss', ':.4e')
    model.train()
    acc_all = AverageMeter('ACC', ':.4e')
    # start_time = time.perf_counter()
    for idx, batch in tqdm(enumerate(loader),total=len(loader),desc=f'Training epoch{epoch}'):
        inputs, labels = batch['image'].float().cuda(), batch['label'].float().cuda()
        text = batch['text']
        processed = [[{'from': d['from'][0], 'value': d['value'][0]} for d in text]]
        optimizer.zero_grad()
        with autocast():
            llamaloss, acc, preds = model(inputs, processed)

        loss = llamaloss
        # loss.backward()  # 缩放损失并反向传播
        # optimizer.step() # 更新参数
        scaler.scale(loss).backward()     # 放大梯度
        scaler.step(optimizer)            # 更新权重
        scaler.update()                   # 更新 scaler

        train_total_loss.update(loss.item())
        acc_all.update(acc)
        # print(
        #         "Train epoch [{}/{}]({}/{}): ".format(epoch, args.max_epochs, idx, len(loader)),
        #         "loss: {:.4f}".format(train_total_loss.avg),
        #         "acc: {:.4f}".format(acc),
        #         "time {:.2f}s".format(time.perf_counter() - start_time),
        #     )
        # start_time = time.perf_counter()
    print(
            "Train epoch [{}/{}]: ".format(epoch, args.max_epochs),
            "loss: {:.4f}".format(train_total_loss.avg),
            "acc: {:.4f}".format(acc_all.avg),
            # "time {:.2f}s".format(time.perf_counter() - start_time),
        )
    if scheduler is not None:
        scheduler.step()
    return train_total_loss.avg


# def val_epoch(data_loader, model, loss_func, metric, epoch, args, save_folder=None, patch_shape = 96):
#     # Setup
#     losses = AverageMeter('Loss', ':.4e')
    
#     metrics = []
#     acc_all = []
    
#     start_time = time.perf_counter()
#     for idx, val_data in enumerate(data_loader):

#         model.eval()
#         with torch.no_grad():
#             val_inputs = val_data["image"].cuda()
#             val_labels = val_data["label"].cuda()
#             text = val_data["text"]
#             processed = [[{'from': d['from'][0], 'value': d['value'][0]} for d in text]]
#             llamaloss, acc, val_outputs = model(val_inputs, processed)
#             #loss_ = loss_func(val_outputs, val_labels, is_train=False)
#             loss = llamaloss #+ loss_
            

#         losses.update(loss.item())
#         acc_all.append(acc)
#         metric_ = metric(val_outputs, val_labels)
#         metrics.extend(metric_)
        
#         print(
#             "Valid epoch [{}/{}]({}/{}): ".format(epoch, args.max_epochs, idx, len(data_loader)),
#             "loss: {:.4f}".format(losses.avg),
#             "time {:.2f}s".format(time.perf_counter() - start_time),
#         )
#         start_time = time.perf_counter()
#     avg_acc = np.mean(acc_all)
#     class_avg_metrics = cal_metrics(metrics, avg_acc, losses.avg, epoch, save_folder)

#     return losses.avg, avg_acc, np.mean(class_avg_metrics)
# bleu = Bleu()
bertscore = evaluate.load("./evaluatemain/metrics/bertscore/bertscore.py")
# rouge = evaluate.load("./evaluatemain/metrics/rouge/rouge.py")
meteor = evaluate.load("./evaluatemain/metrics/meteor/meteor.py")


def val_epoch(data_loader, model, epoch, save_folder=None):
    # start_time = time.perf_counter()
    input = "Is there any anomaly in this brainMRI?"
    describles = "This is a brain MRI image intended for anomaly detection. The image should be free from any signs of pathology, without tumors, lesions, hemorrhages, or other abnormalities."
    prompt = describles + ' ' + input
    prompt_text = f'{prompt}'
    result = dict()
    # bleu_all = 0
    # rouge1_all = 0
    meteor_all= 0
    bert_f1_all = 0
    n=0
    for idx, val_data in tqdm(enumerate(data_loader),total=len(data_loader),desc=f'val epoch{epoch}'):
        n += 1
        model.eval()
        with torch.no_grad():
            val_inputs = val_data["image"].cuda()
            # val_labels = val_data["label"].cuda()
            text = val_data["text"]
            decoded_labels = [d['value'][0] for d in text if d['from'][0] == 'gpt']
            try:
                response, _ = model.generate({
                        'prompt': prompt_text,
                        'image_paths': val_inputs,
                        'top_p': 0.9,
                        'temperature': 1.0,
                        'max_tgt_len': 128,
                    })
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                print(f"Skipping batch due to error: {e}")
                torch.cuda.empty_cache()
                continue
            decoded_preds = [response.strip()]
            try:
                # bleu_score = bleu.compute(predictions=decoded_preds, references=[decoded_labels], max_order=1)
                # bleu_all += bleu_score['bleu']

                # rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels, rouge_types=['rouge1'])
                # rouge1_all += rouge_score['rouge1']

                meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_labels)
                meteor_all += meteor_score['meteor']

                bert_score = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
                bert_f1_all += sum(bert_score['f1']) / len(bert_score['f1'])
            except Exception as e:
                print(f"警告: 样本 {idx} 计算错误 {e}")
                continue
            del val_inputs, response, decoded_preds, decoded_labels, text
            torch.cuda.empty_cache()
                    
    # result["bleu"] = bleu_all / n
    # result["rouge1"] = rouge1_all / n
    result["meteor"] = meteor_all / n
    result["bert_f1"] = bert_f1_all / n
    cal_save(result, epoch, save_folder, 'validation')
    return result


def run_training(
        model,
        start_epoch,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_func,
        eval_metric,
        scaler,
        args,
    ):
    print("Start training ...")

    best_meteor ,best_bert_f1 = 0.0, 0.0
    best_epoch = 0
    for epoch in range(start_epoch + 1, args.max_epochs+1):
        train_epoch(epoch, model, train_loader, optimizer, scheduler, loss_func, scaler, args)

        # Validate at the end of epoch every eval interval
        if epoch % args.eval_interval == 0:

            result = val_epoch(val_loader, model, epoch,save_folder=args.ckpt_save_dir)
            
            # print(
            #     "Valid epoch [{}/{}]: ".format(epoch, args.max_epochs),
            #     "bleu: {:.2f}".format(result["bleu"]),
            #     "rouge1: {:.2f}".format(result["rouge1"]),
            #     "meteor: {:.2f}".format(result["meteor"]),
            #     "bert_f1: {:.2f}".format(result["bert_f1"]),
            # )
            
            # val_acc = (result["bleu"] + result["rouge1"] + result["meteor"] + result["bert_f1"])/4
            bert_f1 = result["bert_f1"]
            meteor = result["meteor"]
            print(meteor,bert_f1)

            if meteor > best_meteor or bert_f1 > best_bert_f1:
                if meteor > best_meteor:
                    best_meteor = meteor
                    print(f"Saving {epoch} epoch with best  {meteor}")
                if bert_f1 > best_bert_f1:
                    best_bert_f1 = bert_f1
                    print(f"Saving {epoch} epoch with best  {bert_f1}")
                save_checkpoint(
                    dict(
                        epoch=epoch,
                        state_dict=model.state_dict(),
                        optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),
                    ), 
                    save_folder=args.ckpt_save_dir, best = True)
    return best_bert_f1

