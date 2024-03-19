#!/bin/bash
# export PSI_SCRATCH="/home/lizz_lab/cse30019698/tmp"
# 获取参数作为工作目录，如果参数为空则设置默认值
if [ -z "$1" ]; then
  work_dir="/home/lizz_lab/cse30019698/project/colmena/multisite_/finetuning-surrogates/runs"
else
  work_dir="$1"
fi

# delete psi4 temp files
rm -r /tmp/psi*
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
python run_test_complex.py \
  --ml-endpoint db55e9cc-ec32-47d6-a6ff-ecd45776d276 \
  --qc-endpoint 698fba9a-4b12-4e0b-b83a-be6ded509946 \
  --training-set ../data/forcefields/starting-model/initial-database.db \
  --search-space ../data/forcefields/starting-model/initial-database.db \
  --starting-model ../data/forcefields/starting-model/starting-model \
  --num-qc-workers 8 \
  --min-run-length 200 \
  --max-run-length 2000 \
  --num-frames 100 \
  --num-epochs 1 \
  --ensemble-size 3 \
  --huber-deltas 1 10 \
  --infer-chunk-size 400 \
  --infer-pool-size 1 \
  --retrain-freq 4 \
  --num-to-run 36 \
  --parsl \
  --no-proxies \
  --redisport 7485 \
  --calculator dft \
  --train-ps-backend redis \
  --sampling-on-device gpu \
  --work-dir "$work_dir" \
  --threads 8 \

echo "Completed at $(date +%Y%m%d_%H%M%S)" 

# delete psi4 temp files
rm -r /tmp/psi*

# cleanning and kill
find /tmp -user $USER -exec mv -t /home/lizz_lab/cse30019698/tmp {} +
echo "tmp file move to ~/tmp, please check and remove them" >> $log_file

user_job_id=$(squeue -u $USER -o "%.18i" | grep -v JOBID | awk '{print $1}')
echo "current task ID is:$user_job_id"
scancel $user_job_id
