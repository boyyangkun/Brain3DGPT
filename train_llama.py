import argparse
import os

import torch
import torch.optim
from torch.cuda.amp import GradScaler
from train_utils import init_config, set_seed
from trainer import run_training
from dataset.multimodal_dataset import get_datasets
from dataset.transforms import *
from model.b3dgpt.openllama import OpenLLAMAPEFTModel
from loss.dice import EDiceLoss
from monai.data import DataLoader
from monai.data.utils import pad_list_data_collate

parser = argparse.ArgumentParser(description='llama downstream segmentation finetune')
parser.add_argument('--start_epoch', default=0, type=int, help='epoch where start training')
parser.add_argument('--max_epochs', default=300, type=int, help='total number of training epoch')

parser.add_argument('--eval_interval', default=10, type=int, help="epoch interval to run validation")
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--patch_shape', default=96, type=int, help='input shape, default to 96')
parser.add_argument('--in_channels', default=4, type=int, help="channels of model inputs")
parser.add_argument('--out_classes', default=3, type=int, help="channels of model outputs")
parser.add_argument('--resume', default='', type=str, help='path of checkpoint to resume from')
parser.add_argument('--pretrained', default='', type=str, help="Pretrained weight path")
parser.add_argument('--mix_template', default=False, type=bool, help='whether to apply random channel substitute with template, default to False')
parser.add_argument('--template_dir', default='', type=str, help='directory to template images (used when mix_template is true)')
parser.add_argument('--use_cl', default=False, type=bool, help='whether to apply contrastive loss, default to False')
parser.add_argument('--cl_weight', default=0.5, type=float, help="train sub")
parser.add_argument('--lr', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--wd', default=0.00001, type=float, help='weight decay')
parser.add_argument('--eta_min', default=0, type=float, help='minimum learning rate, default to 0')

parser.add_argument('--workers', default=8, type=int, help='number of workers, default to 8')                    
parser.add_argument('--devices', default='0', type=str, help='cuda visible devices, default to 0')
parser.add_argument('--random_seed', type=int,default=42, help='random seed')
parser.add_argument("--local_rank", type=int, default=0, help="local rank")

parser.add_argument('--dataset', type=str, default='upenngbm', choices=['brats18', 'brats2021', 'brats23-ped', 'brats23-met',
                                                                        'isles22','mrbrains13', 'vsseg', 'upenngbm'])
parser.add_argument('--data_root', default='', type=str, help='root path to images')
parser.add_argument('--json_file', default='dataset.json', type=str, help='json file name')
parser.add_argument('--experiment', default='baseline', type=str, help='exp name')
parser.add_argument('--output_dir', default='runs', type=str, help='output dir')
parser.add_argument('--cfg', type=str, default="configs/config.yaml", help='path to config file')
parser.add_argument('--llama_ckpt_path', default='', type=str, help='path to Qwen checkpoint')
parser.add_argument('--max_tgt_len', default=1024, type=int,help='the maximum sequence length')
parser.add_argument('--lora_r', default=16, type=int)
parser.add_argument('--lora_alpha', default=64, type=int)
parser.add_argument('--lora_dropout', default=0.05, type=float)
parser.add_argument('--delta_ckpt_path', default='', type=str, help='path to pandagpt checkpoint')
parser.add_argument('--stage', type=int, default=2, help='stage')

def main(args):
    
    args.rank = 0
    init_config(args)
    print(args)
    set_seed(args)
    
    # Model initialization.
    print("Building model ...")
    model = OpenLLAMAPEFTModel(args)

    #### Load weights.
    if args.resume:
        resume_ck = torch.load(args.resume, map_location=torch.device('cpu'))
        model.load_state_dict(resume_ck['state_dict'], strict=True)
    else:
        print("Do not load pretrained weight.")
    
    model = model.cuda()
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Optim
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.wd
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epochs, eta_min=args.eta_min)
    if args.resume:
        optimizer.load_state_dict(resume_ck["optimizer"])
        scheduler.load_state_dict(resume_ck["scheduler"])
        
    # Create dataloader.
    train_dataset, val_dataset, _ = get_datasets(args)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn = pad_list_data_collate, pin_memory=torch.cuda.is_available(), drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn = pad_list_data_collate, pin_memory=torch.cuda.is_available())

    print(f'Length of train loader: {len(train_loader)}')
    print(f'Length of validation loader: {len(val_loader)}')

    start_epoch = 0 if not args.resume else resume_ck["epoch"]
    # Loss func
    loss_func = EDiceLoss().cuda()
    eval_metrics = loss_func.metric
    
    scaler = GradScaler()  # 用于混合精度训练
    # Train epoch
    max_val_dice = run_training(model, 
                                start_epoch, 
                                train_loader, 
                                val_loader, 
                                optimizer, 
                                scheduler, 
                                loss_func,
                                eval_metrics,
                                scaler,
                                args)
    return max_val_dice


if __name__ == '__main__':
    arguments = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)
