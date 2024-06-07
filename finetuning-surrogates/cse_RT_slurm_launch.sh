#!/bin/bash
#SBATCH -o job.%j.out          # 脚本执行的输出将被保存在当job.%j.out文件下，%j表示作业号;
#SBATCH --partition=gpulab01      # 作业提交的指定分区队列为titan
#SBATCH --qos=gpulab01            # 指定作业的QOS
#SBATCH -J finetuning-surrogates       # 作业在调度系统中的作业名为myFirstJob;
#SBATCH --nodes=2              # 申请节点数为1,如果作业不能跨节点(MPI)运行, 申请的节点数应不超过1
#SBATCH --ntasks-per-node=56    # 每个节点上运行一个任务，默认一情况下也可理解为每个节点使用一个核心；
#SBATCH --gres=gpu:4           # 指定作业的需要的GPU卡数量，集群不一样，注意最大限制; 
#SBATCH --time=2:00:00       

## description here 
job_desc="test on clester RT, test complex"
current_date=$(date +'%Y%m%d_%H%M%S')
echo "Started at ${current_date}"

## scripts could get curr dir, should placed under ./fituning-surrogates!
# work_dir=$(cd $(dirname $0);pwd) sbatch may copy script to /var, wot work for this command

home_dir=$(cd ~;pwd)
# proj_dir="/home/lizz_lab/cse12232433/project/colmena/multisite_"
proj_dir="${home_dir}/project/colmena/multisite_"
# work_dir="/home/lizz_lab/cse12232433/project/colmena/multisite_/finetuning-surrogates"
work_dir="${proj_dir}/finetuning-surrogates"

run_dir="${work_dir}/runs/${current_date}"
log_file="${work_dir}/runs/${current_date}/yxx.log"
resources_file="${work_dir}/runs/${current_date}/slurm_resources.ini"

mkdir -p $run_dir

# 获取节点列表
node_list=$(scontrol show hostnames $SLURM_JOB_NODELIST)
echo ${node_list[@]} >> $log_file
echo "job_desc=${job_desc}" >> $log_file

# 获得申请的资源
python ${proj_dir}/my_util/get_slurm_info.py --slurm_job_id ${SLURM_JOB_ID}  --slurm_resources_file ${resources_file} >> $log_file

# for node in $node_list; do
#     ssh $node \
#     "cd $work_dir; \
#     redis-server ./redis.conf & \
#     ./csecluster_test.sh ${run_dir} & \
#     ./runs/analysis/monitor.sh &
#     "
# done

mapfile -t node_list <<< "$node_list"
echo ${node_list[@]} >> $log_file
for node in "${node_list[@]}"; do
    echo "$node" >> $log_file

    echo "ssh $node \"
        cd $work_dir;
        nohup ./analysis/monitor.sh $run_dir >> $log_file 2>&1 &
    \"" >> $log_file
    ssh "$node" "
        cd $work_dir;
        nohup ./analysis/monitor.sh $run_dir >> $log_file 2>&1 &
    "
done
# 在第一个节点上运行所有指令
node=${node_list[0]}
    echo "ssh $node \"
        cd $work_dir;
        conda activate multisite;
        redis-server ../redis.conf &
        ./csecluster_test.sh $run_dir &
    \"" >> $log_file
    ssh "$node" "
        cd $work_dir; 
        conda activate multisite;
        redis-server ../redis.conf &
        ./csecluster_test.sh $run_dir &
    "

wait


# cleanning and kill, should execute on each node, we add this logic in csecluster_test.sh
# find /tmp -user $USER -exec mv -t /home/lizz_lab/cse12232433/tmp {} +
# echo "tmp file move to ~/tmp, please check and remove them" >> $log_file
