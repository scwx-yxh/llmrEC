export CUDA_VISIBLE_DEVICES=1 
python main_modal.py \
    --base_model /home/share/yangxuanhui/llama-3.2-1B \
    --data_path kwai \
    --task_type general \
    --output_dir /home/share/yangxuanhui/result/kwai/modal \
    --batch_size 64 \
    --micro_batch_size 32 \
    --num_epochs 100 \
    --seed 2024 \
    --learning_rate 0.0001 \
    --max_len 32 \
    --cutoff_len 4096 \
    --val_set_size 10 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj, k_proj, v_proj, o_proj, gate_proj, down_proj, up_proj]' \
    --train_on_inputs False \
    --add_eos_token False \
    --group_by_length False \
    --prompt_template_name vicuna \
    --lr_scheduler 'cosine' \
    --warmup_steps 100 \
    ##--resume_from_checkpoint /home/share/yangxuanhui/result/kwai/modal/checkpoint-8000