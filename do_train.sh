ulimit -n 65536
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python3 -u train_llama.py --max_epochs 90 --eval_interval 3 \
--batch_size 1 \
--patch_shape 96 \
--in_channels 1 \
--out_classes 1 \
--lr 0.00001 \
--wd 0.0 \
--random_seed 42 \
--workers 6 \
--devices 3 \
--data_root './data' \
--dataset 'brats2021' \
--json_file 'brain.json' \
--output_dir 'debug' \
--experiment 'brain3DGPT' \
--max_tgt_len 512 \
--pretrained '' \
--llama_ckpt_path '' \
--delta_ckpt_path '' \
