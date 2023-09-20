#!/bin/bash
# 获取参数作为工作目录，如果参数为空则设置默认值
if [ -z "$1" ]; then
  save_dir="/home/lizz_lab/cse12232433/project/colmena/multisite_/finetuning-surrogates/runs/monitor_logs"
else
  save_dir="$1"
fi
mkdir -p $save_dir

cpu_filename="${save_dir}/${HOSTNAME}cpu_usage_$(date +%Y%m%d_%H%M%S).log"
gpu_filename="${save_dir}/${HOSTNAME}gpu_usage_$(date +%Y%m%d_%H%M%S).log"
mem_filename="${save_dir}/${HOSTNAME}mem_usage_$(date +%Y%m%d_%H%M%S).log"


echo "Timestamp,GPUs,GPU Utilization (%)" > "$gpu_filename"
echo "Timestamp    title    total        used        free      shared  buff/cache   available" > "$mem_filename"

# 同时执行mpstat、free和nvidia-smi命令
# 每个log都自己打印时间戳，避免错误和减少数据，2s打印一次
mpstat -P ALL 2 >> "$cpu_filename"  &

# free -s 1 -h | awk -v timestamp="$(date +"%Y-%m-%d_%H:%M:%S")" 'NR==2 {print timestamp "  " $0}' >> "$mem_filename"  &

# nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader -l 2 | awk -v timestamp="$(date +"%Y-%m-%d_%H:%M:%S")" '{print timestamp "," $1  $2}' >> "$gpu_filename"  &

## 一个小时后自动关闭脚本
# sleep 3600
# kill -9 $(ps -ef | grep monitor.sh | grep -v grep | awk '{print $2}')


# 获取当前时间戳，记录
start_timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
end_timestamp=""
# 设置记录时间为1小时（以秒为单位）
record_time=$((60*60))

# 循环记录数据
while true; do
    # 获取当前时间戳
    timestamp="$(date '+%Y-%m-%d-%H:%M:%S')"
    
    # 获取GPU利用率并写入日志文件
    nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader | awk -v timestamp="$timestamp" '{print timestamp "," $1  $2}' >> "$gpu_filename"
    # mpstat -P ALL 2 1 >> "$cpu_filename"
    memory_info=$(free -h | awk 'NR==2 {print $0}')
    echo "$timestamp $memory_info" >> "$mem_filename"
    
    # 检查是否达到记录时间
    current_timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    seconds_passed=$(( $(date -d "$current_timestamp" '+%s') - $(date -d "$start_timestamp" '+%s') ))
    
    if [ $seconds_passed -ge $record_time ]; then
        # 达到记录时间，记录结束时间并退出循环
        end_timestamp="$current_timestamp"
        break
    fi
    
    sleep 2
done

kill $(pgrep mpstat)
kill $(pgrep free)
kill $(pgrep nvidia-smi)

echo "monitor.sh ended"