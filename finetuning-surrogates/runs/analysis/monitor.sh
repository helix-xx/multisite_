#!/bin/bash

save_dir="/home/lizz_lab/cse12232433/project/colmena/multisite_/finetuning-surrogates/runs/analysis/monitor_logs"
cpu_filename="${save_dir}/${HOSTNAME}cpu_usage_$(date +%Y%m%d_%H%M%S).log"
gpu_filename="${save_dir}/${HOSTNAME}gpu_usage_$(date +%Y%m%d_%H%M%S).log"
mem_filename="${save_dir}/${HOSTNAME}mem_usage_$(date +%Y%m%d_%H%M%S).log"
# mpstat -P ALL 1 10 >> "$cpu_filename"
# while true; do
#     nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader >> "$gpu_filename"
#     sleep 1
# done

# 创建并写入GPU利用率日志文件的标题行
# echo $HOSTNAME > "$cpu_filename"
# echo $HOSTNAME > "$gpu_filename"
echo "Timestamp,GPUs,GPU Utilization (%)" > "$gpu_filename"

# 同时执行mpstat、free和nvidia-smi命令
# 时间戳通过mpstat获得
# mpstat -P ALL 1 >> "$cpu_filename"  &

# free -s 1 -h >> "$mem_filename"  &

# nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader -l 1 | awk -v timestamp="$(date +"%Y-%m-%d %H:%M:%S")" '{print timestamp "," $1  $2}' >> "$gpu_filename"  &

# ## 一个小时后自动关闭脚本
# sleep 3600
# kill -9 $(ps -ef | grep monitor.sh | grep -v grep | awk '{print $2}')


## 下面这个程序记录时间不准确，使用内置的每秒打印一次的参数,或只根据mpstat的时间戳，记录数据总量是一致的
# while true; do
#     # 获取当前时间戳
#     timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    
#     # 获取GPU利用率并写入日志文件
#     nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | awk -v timestamp="$timestamp" '{print timestamp "," $1  $2}' >> "$gpu_filename"
#     mpstat -P ALL >> "$cpu_filename"
#     free -h >> "$mem_filename"
    
#     # 等待1秒
#     sleep 1
# done

# 获取当前时间戳，记录
start_timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
end_timestamp=""
# 设置记录时间为1小时（以秒为单位）
record_time=$((60 * 60))

# 循环记录数据
while true; do
    # 获取当前时间戳
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    
    # 获取GPU利用率并写入日志文件
    nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | awk -v timestamp="$timestamp" '{print timestamp "," $1  $2}' >> "$gpu_filename"
    mpstat -P ALL >> "$cpu_filename"
    free -h >> "$mem_filename"
    
    # 检查是否达到记录时间
    current_timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    seconds_passed=$(( $(date -d "$current_timestamp" '+%s') - $(date -d "$start_timestamp" '+%s') ))
    
    if [ $seconds_passed -ge $record_time ]; then
        # 达到记录时间，记录结束时间并退出循环
        end_timestamp="$current_timestamp"
        break
    fi
    
    # 等待1秒
    sleep 1
done