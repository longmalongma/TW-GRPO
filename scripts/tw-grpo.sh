export PRIVATE_DATA_ROOT=TW-GRPO
export WANDB_PROJECT=Qwen2.5-VL-7B-Video-GRPO
export MODEL_NAME=Qwen2.5-VL-7B-Instruct_clevrer_counterfactual_twgrpo_with_alpha17
export WANDB_NAME=$MODEL_NAME
export DEBUG_MODE=true
export LOG_PATH=$PRIVATE_DATA_ROOT/$WANDB_PROJECT/$WANDB_NAME/debug.log
export WANDB_MODE=offline
export SAMPLE_MODE=true
export LOG_PATH=$PRIVATE_DATA_ROOT/$WANDB_NAME/debug.log

torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12532" \
    src/open_r1/grpo.py \
    --deepspeed scripts/zero3_offload.json \
    --output_dir $PRIVATE_DATA_ROOT/$MODEL_NAME \
    --model_name_or_path $PRIVATE_DATA_ROOT/Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset_name xxx \
    --jsonl_path $PRIVATE_DATA_ROOT/data/CLEVERER/clevrer_counterfactual_train.json \
    --max_prompt_length 4096 \
    --max_completion_length 4096 \
    --reward_funcs accuracy format \
    --learning_rate 1e-6 \
    --beta 0.00 \
    --alpha 1.70 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --question_type mixed \
    --bf16 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --freeze_vision_modules true \
    --loss_type tw_grpo \
    --num_train_epochs 1 \
    --run_name $WANDB_NAME \
    --save_steps 100 \
    --max_grad_norm 20 \
    --save_only_model true \
    --num_generations 8