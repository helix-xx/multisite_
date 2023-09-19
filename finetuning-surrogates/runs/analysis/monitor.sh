#!/bin/bash

save_dir="/home/yxx/work/project/colmena/multisite_/finetuning-surrogates/runs/monitor_logs"
cpu_filename="${save_dir}/cpu_usage_$(date +%Y%m%d_%H%M%S).log"
gpu_filename="${save_dir}/gpu_usage_$(date +%Y%m%d_%H%M%S).log"
# mpstat -P ALL 1 10 >> "$cpu_filename"
# while true; do
#     nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader >> "$gpu_filename"
#     sleep 1
# done

# 创建并写入GPU利用率日志文件的标题行
echo "Timestamp,GPU Utilization (%)" > "$gpu_filename"

# 同时执行mpstat和nvidia-smi命令
mpstat -P ALL 1 >> "$cpu_filename" >> "$cpu_filename" &

while true; do
    # 获取当前时间戳
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    
    # 获取GPU利用率并写入日志文件
    nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | awk -v timestamp="$timestamp" '{print timestamp "," $2}' >> "$gpu_filename"
    
    # 等待1秒
    sleep 1
done