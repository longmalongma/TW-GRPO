export MODEL_NAME=Video-R1/Qwen2.5-VL-7B-COT-SFT

echo "Training completed. Starting evaluation..."
python src/eval/eval_clevrer.py \
    --model_name $MODEL_NAME \
    --batch_size 16

echo "Training completed. Starting evaluation..."
python src/eval/eval_general_videor1.py \
    --model_name $MODEL_NAME \
    --dataset_name eval_nextgqa \
    --batch_size 16

echo "Training completed. Starting evaluation..."
python src/eval/eval_general_videor1.py \
    --model_name $MODEL_NAME \
    --dataset_name eval_mmvu \
    --batch_size 16

echo "Training completed. Starting evaluation..."
python src/eval/eval_general_videor1.py \
    --model_name $MODEL_NAME \
    --dataset_name eval_mvbench \
    --batch_size 16

echo "Training completed. Starting evaluation..."
python src/eval/eval_general_videor1.py \
    --model_name $MODEL_NAME \
    --dataset_name eval_tempcompass \
    --batch_size 16

echo "Training completed. Starting evaluation..."
python src/eval/eval_general_videor1.py \
    --model_name $MODEL_NAME \
    --dataset_name eval_videomme \
    --batch_size 16
