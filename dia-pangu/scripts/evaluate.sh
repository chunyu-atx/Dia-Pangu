export ASCEND_VISIBLE_DEVICES=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export NLTK_DATA=/media/t1/zcy/dia-pangu/evaluate/nltk_data
echo "--- NLTK_DATA environment variable set to: $NLTK_DATA ---"
torchrun --nproc_per_node=1 --master_port=25382 /media/t1/zcy/dia-pangu/evaluate/Evaluate.py