#!/bin/bash
#SBATCH -o /home/lizz_lab/cse12232433/project/colmena/multisite_/finetuning-surrogates/job_out/perf_collection.%j.out
#SBATCH --partition=gpulab01
#SBATCH --qos=gpulab01
#SBATCH -J perf_collection
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=40
#SBATCH --gres=gpu:4
#SBATCH --time=4:00:00

# 描述
job_desc="Performance data collection for task scheduling"
current_date=$(date +'%Y%m%d_%H%M%S')
echo "Started at ${current_date}"

# 设置目录
home_dir=$(cd ~;pwd)
proj_dir="${home_dir}/project/colmena/multisite_"
work_dir="${proj_dir}/finetuning-surrogates"

run_dir="${work_dir}/runs/${current_date}"
log_file="${work_dir}/runs/${current_date}/perf_collection.log"
resources_file="${work_dir}/runs/${current_date}/slurm_resources.ini"

mkdir -p $run_dir

# 获取节点信息
node_list=$(scontrol show hostnames $SLURM_JOB_NODELIST)
echo ${node_list[@]} >> $log_file
echo "job_desc=${job_desc}" >> $log_file

# 获取资源信息
python ${proj_dir}/my_util/get_slurm_info.py --slurm_job_id ${SLURM_JOB_ID} --slurm_resources_file ${resources_file} >> $log_file

# 启动监控
mapfile -t node_list <<< "$node_list"
echo ${node_list[@]} >> $log_file
for node in "${node_list[@]}"; do
    echo "$node" >> $log_file
    ssh "$node" "
        cd $work_dir;
        nohup ./analysis/monitor.sh $run_dir >> $log_file 2>&1 &
    "
done

# 在第一个节点上运行主程序
node=${node_list[0]}
echo "ssh $node \"
    cd $work_dir;
    conda activate multisite;
    redis-server ../redis.conf &
    python run_performance_collection.py \
        --task-queue-path /home/lizz_lab/cse12232433/project/colmena/multisite_/finetuning-surrogates/runs/hist_data/task_queue_audit.pkl \
        --work-dir $run_dir \
        --redisport 7485 \
        --cpu-configs 1 2 4 8 12 16 20 \
        --gpu-configs  1 2 3 4 \
        --samples-per-config 2 \
        --starting-model ../data/forcefields/starting-model/starting-model &
\"" >> $log_file

ssh "$node" "
    cd $work_dir;
    conda activate multisite;
    redis-server ../redis.conf &
    python run_performance_collection.py \
        --task-queue-path /home/lizz_lab/cse12232433/project/colmena/multisite_/finetuning-surrogates/runs/hist_data/task_queue_audit.pkl \
        --work-dir $run_dir \
        --redisport 7485 \
        --cpu-configs 1 2 4 8 12 16 20 \
        --gpu-configs  1 2 3 4 \
        --samples-per-config 2 \
        --starting-model ../data/forcefields/starting-model/starting-model &
"

wait

# 清理临时文件
rm -r /tmp/psi*
find /tmp -user $USER -exec mv -t /home/lizz_lab/cse12232433/tmp {} +
echo "tmp file move to ~/tmp, please check and remove them" >> $log_file

# 结束作业
user_job_id=$(squeue -u $USER -o "%.18i" | grep -v JOBID | awk '{print $1}')
echo "current task ID is:$user_job_id"
scancel $user_job_id