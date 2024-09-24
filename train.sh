CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.train \
  --output_dir model_msmarco_passage_multiquery \
  --model_name_or_path bert-base-uncased \
  --save_steps 20000 \
  --dataset_path /workspaces/tevatron/data/data.jsonl \
  --query_prefix "Query: " \
  --passage_prefix "Passage: " \
  --fp16 \
  --per_device_train_batch_size 32 \
  --train_group_size 1 \
  --dataloader_num_workers 1 \
  --learning_rate 4e-5 \
  --query_max_len 128 \
  --passage_max_len 512 \
  --normalize \
  --temperature 0.01 \
  --num_train_epochs 1 \
  --logging_steps 500 \
  --overwrite_output_dir \
  --loss_type symmetric \
  --dataset_type passage_multiquery