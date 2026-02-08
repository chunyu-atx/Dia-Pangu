experiment_name="dia-pangu_v0112"
export ASCEND_VISIBLE_DEVICES=0

torchrun --nproc_per_node=1 --master_port=25380 /media/t1/zcy/dia-pangu/src/train.py \
    --bf16 True \
    --lang_encoder_path "/media/t1/zcy/openPangu-Embedded-7B-V1.1" \
    --tokenizer_path "/media/t1/zcy/openPangu-Embedded-7B-V1.1" \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 8 \
    --save_strategy "epoch" \
    --save_steps 1000 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_steps 20 \
    --lr_scheduler_type "cosine" \
    --dataloader_num_workers 8 \
    --run_name $experiment_name \
    --output_dir "/media/t1/zcy/dia-pangu/checkpoint_new/$experiment_name" \
    --logging_steps 1
