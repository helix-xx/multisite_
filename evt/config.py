from parsl.executors import HighThroughputExecutor, WorkQueueExecutor
from parsl.providers import CobaltProvider, AdHocProvider, SlurmProvider, LocalProvider
from parsl.addresses import address_by_hostname
from parsl.launchers import AprunLauncher,SrunLauncher
from parsl.channels import SSHChannel, LocalChannel, SSHInteractiveLoginChannel
from parsl import Config

###################################################################
# csecluster config
###################################################################
def csecluster1(log_dir: str) -> Config:
    config = Config(
        run_dir=log_dir,
        strategy='none',
        retries=1,
        executors=[
            HighThroughputExecutor(
                label="cpu",
                max_workers=8,
                address=address_by_hostname(),
                worker_port_range=(20000,30000),
                worker_logdir_root='/home/lizz_lab/cse30019698/parsl-logs',
                provider=LocalProvider(
                    # min_blocks=1,
                    # max_blocks=4,
                    init_blocks=4,
                    # channels=[SSHChannel(hostname='gpu001',port='22', username='cse12232433', password='Yxx!199871!', script_dir='/home/lizz_lab/cse12232433/parsl-logs')],
                    channel=LocalChannel(),
                    worker_init='''
                    # Activate conda environment
                    source /home/lizz_lab/cse30019698/.bashrc
                    source /home/lizz_lab/cse30019698/software/miniconda3/bin/activate /home/lizz_lab/cse30019698/software/miniconda3/envs/multisite
                    # export CUDA_VISIBLE_DEVICES=0,1,2,3
                    which python
                    ''',
                ),
            ),
            HighThroughputExecutor(
                address=address_by_hostname(),
                label="gpu",
                available_accelerators=4,
                max_workers=1,
                cores_per_worker=1,
                worker_port_range=(20000,30000),
                worker_logdir_root='/home/lizz_lab/cse30019698/parsl-logs',
                provider=LocalProvider(
                    # min_blocks=1,
                    # max_blocks=4,
                    init_blocks=4,
                    # channels=[SSHChannel(hostname='gpu001',port='22', username='cse12232433', password='Yxx!199871!', script_dir='/home/lizz_lab/cse12232433/parsl-logs')],
                    channel=LocalChannel(),
                    worker_init='''
                    # Activate conda environment
                    source /home/lizz_lab/cse30019698/.bashrc
                    source /home/lizz_lab/cse30019698/software/miniconda3/bin/activate /home/lizz_lab/cse30019698/software/miniconda3/envs/multisite
                    # bash /home/lizz_lab/cse30019698/project/colmena/multisite_/finetuning-surrogates/runs/analysis/monitor.sh &
                    # export TF_GPU_ALLOCATOR=cuda_malloc_async
                    which python
                    ''',
                ),
            )]
    )
        
    return config