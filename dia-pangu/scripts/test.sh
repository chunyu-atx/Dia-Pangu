export ASCEND_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 --master_port=25380 /media/t1/zcy/dia-pangu/src/test.py