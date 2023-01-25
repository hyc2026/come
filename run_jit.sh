# date=`date +%s`
date=qt
mkdir -p exp/$date/cache/s1
mkdir -p exp/$date/cache/s2
mkdir -p exp/$date/cache/retrieval
mkdir -p exp/$date/prediction/s1
mkdir -p exp/$date/prediction/s2
mkdir -p exp/$date/prediction/valid
mkdir -p exp/$date/output/s1
mkdir -p exp/$date/output/s2
mkdir -p exp/$date/output/retrieval
mkdir -p summary/s1
mkdir -p summary/s1

batch_size=16
max_source_length=512
max_target_length=100
output_dir=/root/CodeT51/exp/$date/output
res_dir=/root/CodeT51/exp/$date/prediction
cache_path=/root/CodeT51/exp/$date/cache
task=qt
learning_rate=3e-5

CUDA_VISIBLE_DEVICES=0 \
  python run_gen.py \
  --do_train --do_eval \
  --task jit --sub_task $task --model_type codet5 --data_num -1 \
  --data_type s1 --num_train_epochs 5 --warmup_steps 1000 --learning_rate $learning_rate \
  --tokenizer_name=Salesforce/codet5-base --model_name_or_path=Salesforce/codet5-base --data_dir /root/CodeT51/data \
  --cache_path $cache_path/s1 --output_dir $output_dir/s1 --summary_dir /root/CodeT51/summary/s1 \
  --save_last_checkpoints --always_save_model --res_dir $res_dir/s1 \
  --train_batch_size $batch_size --eval_batch_size $batch_size --max_source_length $max_source_length --max_target_length $max_target_length

cp /root/CodeT51/config.json /root/CodeT51/exp/$date/output/s1/checkpoint-best-ppl/

CUDA_VISIBLE_DEVICES=0 \
  python run_gen.py \
  --do_jit \
  --task jit --sub_task $task --model_type codet5 --data_num -1 \
  --data_type s2 --num_train_epochs 10 --warmup_steps 1000 --learning_rate $learning_rate \
  --tokenizer_name=Salesforce/codet5-base --model_name_or_path=/root/CodeT51/exp/$date/output/s1/checkpoint-best-ppl --data_dir /root/CodeT51/data \
  --cache_path $cache_path/s2 --output_dir $output_dir/s2 --summary_dir /root/CodeT51/summary/s2 \
  --save_last_checkpoints --always_save_model --res_dir $res_dir/s2 \
  --train_batch_size $batch_size --eval_batch_size $batch_size --max_source_length $max_source_length --max_target_length $max_target_length
