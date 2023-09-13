#!/bin/bash

save_dir="/home/lizz_lab/cse12232433/project/colmena/multisite_/finetuning-surrogates/monitor_logs"
cpu_filename="${save_dir}/cpu_usage_$(date +%Y%m%d_%H%M%S).log"
gpu_filename="${save_dir}/gpu_usage_$(date +%Y%m%d_%H%M%S).log"
mpstat -P ALL 1 10 >> "$cpu_filename"
while true; do
    nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader >> "$gpufilename"
    sleep 1
done