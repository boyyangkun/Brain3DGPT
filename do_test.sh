ulimit -n 65536
python3 test.py \
--devices 0 \
--batch_size 1 \
--in_channels 1 \
--out_classes 1 \
--patch_shape 96 \
--dataset upenngbm \
--data_root './data' \
--json_file 'brain.json' \
--checkpoint '' \
--max_tgt_len 128 \
--pretrained '' \
--Qwen_ckpt_path '' \
--delta_ckpt_path ''
