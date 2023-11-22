#!/bin/bash

# 获取参数作为工作目录，如果参数为空则设置默认值
if [ -z "$1" ]; then
  work_dir="/home/lizz_lab/cse30019698/project/colmena/multisite_/finetuning-surrogates/runs"
else
  work_dir="$1"
fi
# Test for the local system
# python run_test.py \
#     --ml-endpoint db55e9cc-ec32-47d6-a6ff-ecd45776d276 \
#     --qc-endpoint 698fba9a-4b12-4e0b-b83a-be6ded509946 \
#     --training-set ../data/forcefields/starting-model/initial-database.db \
#     --search-space ../data/forcefields/starting-model/initial-database.db \
#     --starting-model ../data/forcefields/starting-model/starting-model \
#     --num-qc-workers 14 \
#     --min-run-length 200 \
#     --max-run-length 2000 \
#     --num-frames 100 \
#     --num-epochs 128 \
#     --ensemble-size 12 \
#     --huber-deltas 1 10 \
#     --infer-chunk-size 200 \
#     --infer-pool-size 1 \
#     --retrain-freq 16 \
#     --num-to-run 30 \
#     --parsl \
#     --no-proxies \
#     --redisport 7485 \
#     --calculator dft \
#     --train-ps-backend redis \
#     --sampling-on-device gpu \
#     --work-dir "$work_dir" \
#     --threads 4 \
#     --redishost gpu005


    ## epoches 512->128->1(test)
    
    ## redisport 7485->7486(test)

    ## baseline
python run_test.py \
  --ml-endpoint db55e9cc-ec32-47d6-a6ff-ecd45776d276 \
  --qc-endpoint 698fba9a-4b12-4e0b-b83a-be6ded509946 \
  --training-set ../data/forcefields/starting-model/initial-database.db \
  --search-space ../data/forcefields/starting-model/initial-database.db \
  --starting-model ../data/forcefields/starting-model/starting-model \
  --num-qc-workers 8 \
  --min-run-length 200 \
  --max-run-length 2000 \
  --num-frames 100 \
  --num-epochs 16 \
  --ensemble-size 2 \
  --huber-deltas 1 10 \
  --infer-chunk-size 4000 \
  --infer-pool-size 1 \
  --retrain-freq 16 \
  --num-to-run 20 \
  --parsl \
  --no-proxies \
  --redisport 7485 \
  --calculator dft \
  --train-ps-backend redis \
  --sampling-on-device gpu \
  --work-dir "$work_dir" \
  --threads 8 \