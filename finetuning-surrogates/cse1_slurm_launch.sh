#!/bin/bash
#SBATCH -o job.%j.out          # 脚本执行的输出将被保存在当job.%j.out文件下，%j表示作业号;
#SBATCH --partition=titan      # 作业提交的指定分区队列为titan
#SBATCH --qos=titan            # 指定作业的QOS
#SBATCH -J finetuning-surrogates       # 作业在调度系统中的作业名为myFirstJob;
#SBATCH --nodes=1              # 申请节点数为1,如果作业不能跨节点(MPI)运行, 申请的节点数应不超过1
#SBATCH --ntasks-per-node=64    # 每个节点上运行一个任务，默认一情况下也可理解为每个节点使用一个核心；
#SBATCH --gres=gpu:4           # 指定作业的需要的GPU卡数量，集群不一样，注意最大限制; 
#SBATCH --time=10:00:00       

## description here 
job_desc="test on cse1, test complex workflow" 

echo "Started at $(date +'%Y%m%d_%H%M%S')"
proj_dir="/home/lizz_lab/cse30019698/project/colmena/multisite_"
work_dir="/home/lizz_lab/cse30019698/project/colmena/multisite_/finetuning-surrogates"
current_date=$(date +'%Y%m%d_%H%M%S')
run_dir="${work_dir}/runs/${current_date}"
log_file="${work_dir}/runs/${current_date}/yxx.log"
mkdir -p $run_dir

# 获取节点列表
node_list=$(scontrol show hostnames $SLURM_JOB_NODELIST)
echo ${node_list[@]} >> $log_file
echo "job_desc=${job_desc}" >> $log_file

# for node in $node_list; do
#     ssh $node \
#     "cd $work_dir; \
#     redis-server ./redis.conf & \
#     ./csecluster_test.sh ${run_dir} & \
#     ./runs/analysis/monitor.sh &
#     "
# done

echo "node=${node_list[0]} \
ssh $node \
\"cd $proj_dir; \
conda activate multisite; \
redis-server ./redis.conf &; \
cd $work_dir; \
./analysis/monitor.sh ${run_dir} &;\
./csecluster_test.sh ${run_dir} &> /dev/null \"" >> $log_file


# node=${node_list[0]}
# ssh $node \
# "cd $proj_dir ; \
# conda activate multisite ; \
# redis-server ./redis.conf & ; \
# cd $work_dir ; \
# ./runs/analysis/monitor.sh ${run_dir} & ; \
# ./csecluster_test.sh ${run_dir} &> /dev/null "

node=${node_list[0]}
ssh $node \
    "cd $proj_dir ; \
    conda activate multisite ; \
    redis-server ./redis.conf & \
    cd $work_dir ; \
    ./analysis/monitor.sh ${run_dir} & \
    ./csecluster_test.sh ${run_dir}  "

echo "Completed at $(date +%Y%m%d_%H%M%S)" >> $log_file
