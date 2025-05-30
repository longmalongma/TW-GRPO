export MODEL_NAME=Qwen2.5-VL-7B-Instruct_clevrer_counterfactual_twgrpo_with_alpha17

echo "Training completed. Starting evaluation..."
python src/eval/eval_clevrer.py \
    --model_name $MODEL_NAME/checkpoint-500 \
    --batch_size 16

echo "Training completed. Starting evaluation..."
python src/eval/eval_general_videor1.py \
    --model_name $MODEL_NAME/checkpoint-500 \
    --dataset_name eval_nextgqa \
    --batch_size 16

echo "Training completed. Starting evaluation..."
python src/eval/eval_general_videor1.py \
    --model_name $MODEL_NAME/checkpoint-500 \
    --dataset_name eval_mmvu \
    --batch_size 16

echo "Training completed. Starting evaluation..."
python src/eval/eval_general_videor1.py \
    --model_name $MODEL_NAME/checkpoint-500 \
    --dataset_name eval_mvbench \
    --batch_size 16

echo "Training completed. Starting evaluation..."
python src/eval/eval_general_videor1.py \
    --model_name $MODEL_NAME/checkpoint-500 \
    --dataset_name eval_tempcompass \
    --batch_size 16

echo "Training completed. Starting evaluation..."
python src/eval/eval_general_videor1.py \
    --model_name $MODEL_NAME/checkpoint-500 \
    --dataset_name eval_videomme \
    --batch_size 16
