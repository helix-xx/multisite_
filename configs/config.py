from parsl.executors import HighThroughputExecutor, WorkQueueExecutor
from parsl.providers import CobaltProvider, AdHocProvider, SlurmProvider, LocalProvider
from parsl.addresses import address_by_hostname
from parsl.launchers import AprunLauncher,SrunLauncher
from parsl.channels import SSHChannel, LocalChannel, SSHInteractiveLoginChannel
from parsl import Config


def theta_debug_and_lambda(log_dir: str) -> Config:
    """Configuration where simulation tasks run on Theta and ML tasks run on Lambda.

    Args:
        log_dir: Path to store monitoring DB and parsl logs
    Returns:
        (Config) Parsl configuration
    """
    # Set a Theta config for using the KNL nodes with a single worker per node
    config = Config(
        run_dir=log_dir,
        retries=1,
        executors=[
            HighThroughputExecutor(
                label='knl',
                max_workers=1,
                address=address_by_hostname(),
                provider=CobaltProvider(
                    queue='debug-flat-quad',  # Flat has lower utilization, even though xTB is (slightly) faster on cache
                    account='CSC249ADCD08',
                    launcher=AprunLauncher(overrides="-d 256 --cc depth -j 4"),
                    worker_init='''
module load miniconda-3
source activate /lus/theta-fs0/projects/CSC249ADCD08/multi-site-campaigns/env-parsl
which python
''',  # Active the environment
                    nodes_per_block=8,
                    init_blocks=0,
                    min_blocks=0,
                    max_blocks=1,
                    cmd_timeout=300,
                    walltime='00:60:00',
                    scheduler_options='#COBALT --attrs enable_ssh=1:filesystems=theta-fs0,home',
            )),
            HighThroughputExecutor(
                address='localhost',
                label="v100",
                available_accelerators=8,
                worker_ports=(54928, 54875),  # Hard coded to match up with SSH tunnels
                worker_logdir_root='/lambda_stor/homes/lward/multi-site-campaigns/parsl-run/logs',
                provider=AdHocProvider(
                    channels=[SSHChannel('lambda2.cels.anl.gov', script_dir='/lambda_stor/homes/lward/multi-site-campaigns/parsl-run')],
                    worker_init='''
# Activate conda environment
source /homes/lward/miniconda3/bin/activate /home/lward/multi-site-campaigns/parsl-logs
which python
''',
                ),
            )]
    )
        
    return config


def theta_debug_and_venti(log_dir: str) -> Config:
    """Configuration where simulation tasks run on Theta and ML tasks run on Lambda.

    Args:
        log_dir: Path to store monitoring DB and parsl logs
    Returns:
        (Config) Parsl configuration
    """
    # Set a Theta config for using the KNL nodes with a single worker per node
    config = Config(
        run_dir=log_dir,
        retries=1,
        executors=[
            HighThroughputExecutor(
                label='cpu',
                max_workers=1,
                address=address_by_hostname(),
                provider=CobaltProvider(
                    queue='debug-flat-quad',  # Flat has lower utilization, even though xTB is (slightly) faster on cache
                    account='CSC249ADCD08',
                    launcher=AprunLauncher(overrides="-d 256 --cc depth -j 4"),
                    worker_init='''
module load miniconda-3
source activate /lus/theta-fs0/projects/CSC249ADCD08/multi-site-campaigns/env
which python
''',  # Active the environment
                    nodes_per_block=8,
                    init_blocks=0,
                    min_blocks=0,
                    max_blocks=1,
                    cmd_timeout=300,
                    walltime='00:60:00',
                    scheduler_options='#COBALT --attrs enable_ssh=1:filesystems=theta-fs0,home',
            )),
            HighThroughputExecutor(
                address='localhost',
                label="gpu",
                available_accelerators=20,
                worker_ports=(54928, 54875),  # Hard coded to match up with SSH tunnels
                worker_logdir_root='/home/lward/multi-site-campaigns/parsl-logs',
                provider=AdHocProvider(
                    channels=[SSHChannel('lambda5.cels.anl.gov', script_dir='/home/lward/multi-site-campaigns/parsl-logs')],
                    worker_init='''
# Activate conda environment
source /homes/lward/miniconda3/bin/activate /home/lward/multi-site-campaigns/env
which python
''',
                ),
            )]
    )
        
    return config

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
                    init_blocks=64,
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

def csecluster1_wq(log_dir: str) -> Config:
    config = Config(
        run_dir=log_dir,
        strategy='none',
        retries=1,
        executors=[
            WorkQueueExecutor(
                label="cpu",
                # max_workers=8,
                address=address_by_hostname(),
                shared_fs=True,
                source=True,
                autolabel=True,
                autolabel_window=10,
                # worker_port_range=(20000,30000),
                # worker_logdir_root='/home/lizz_lab/cse30019698/parsl-logs',
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
            workQueueExecutor(
                address=address_by_hostname(),
                label="gpu",
                shared_fs=True,
                source=True,
                autolabel=True,
                autolabel_window=10,
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

def csecluster_RT(log_dir: str) -> Config:
    config = Config(
        run_dir=log_dir,
        retries=1,
        executors=[
            HighThroughputExecutor(
                label="cpu",
                max_workers=56,
                address=address_by_hostname(),
                worker_port_range=(20000,30000),
                worker_logdir_root='/home/lizz_lab/cse12232433/parsl-logs',
                provider=AdHocProvider(
                    channels=[SSHChannel(hostname='gpu005',port='22', username='cse12232433', password='Yxx!199871!', script_dir='/home/lizz_lab/cse12232433/parsl-logs'),
                              SSHChannel(hostname='gpu006',port='22', username='cse12232433', password='Yxx!199871!', script_dir='/home/lizz_lab/cse12232433/parsl-logs')],
                    worker_init='''
                    # Activate conda environment
                    source /home/lizz_lab/cse12232433/.bashrc
                    source /home/lizz_lab/cse12232433/miniconda3/bin/activate /home/lizz_lab/cse12232433/miniconda3/envs/multisite
                    # export CUDA_VISIBLE_DEVICES=0,1,2,3
                    # bash /home/lizz_lab/cse12232433/project/colmena/multisite_/finetuning-surrogates/runs/analysis/monitor.sh
                    which python
                    ''',
                ),
            ),
            HighThroughputExecutor(
                address=address_by_hostname(),
                label="gpu",
                available_accelerators=4,
                worker_port_range=(20000,30000),
                worker_logdir_root='/home/lizz_lab/cse12232433/parsl-logs',
                provider=AdHocProvider(
                    channels=[SSHChannel(hostname='gpu005',port='22', username='cse12232433', password='Yxx!199871!', script_dir='/home/lizz_lab/cse12232433/parsl-logs'),
                              SSHChannel(hostname='gpu006',port='22', username='cse12232433', password='Yxx!199871!', script_dir='/home/lizz_lab/cse12232433/parsl-logs')],
                    worker_init='''
                    # Activate conda environment
                    source /home/lizz_lab/cse12232433/.bashrc
                    source /home/lizz_lab/cse12232433/miniconda3/bin/activate /home/lizz_lab/cse12232433/miniconda3/envs/multisite
                    export CUDA_VISIBLE_DEVICES=0,1,2,3
                    # export TF_GPU_ALLOCATOR=cuda_malloc_async
                    # bash /home/lizz_lab/cse12232433/project/colmena/multisite_/finetuning-surrogates/runs/analysis/monitor.sh
                    which python
                    ''',
                ),
            )]
    )
        
    return config


def csecluster_RT_slurm(log_dir: str) -> Config:
    config = Config(
        run_dir=log_dir,
        strategy='none',
        retries=1,
        executors=[
            HighThroughputExecutor(
                label="cpu",
                max_workers=1,
                address=address_by_hostname(),
                worker_port_range=(20000,30000),
                worker_logdir_root='/home/lizz_lab/cse12232433/parsl-logs',
                # provider=LocalProvider(
                #     # min_blocks=1,
                #     # max_blocks=4,
                #     init_blocks=16,
                #     # channels=[SSHChannel(hostname='gpu001',port='22', username='cse12232433', password='Yxx!199871!', script_dir='/home/lizz_lab/cse12232433/parsl-logs')],
                #     channel=LocalChannel(),
                #     worker_init='''
                #     # Activate conda environment
                #     source /home/lizz_lab/cse12232433/.bashrc
                #     source /home/lizz_lab/cse12232433/miniconda3/bin/activate /home/lizz_lab/cse12232433/miniconda3/envs/multisite
                #     # export CUDA_VISIBLE_DEVICES=0,1,2,3
                #     which python
                #     ''',
                # ),
                provider=SlurmProvider(
                    partition = "gpulab01",
                    channel = LocalChannel(
                        envs={},
                        script_dir=None,
                        userhome='/home/lizz_lab/cse12232433/project/colmena/multisite_/finetuning-surrogates/',
                    ),
                    nodes_per_block=1,
                    init_blocks=1,
                    worker_init = '''
                    source /home/lizz_lab/cse12232433/miniconda3/bin/activate /home/lizz_lab/cse12232433/miniconda3/envs/multisite
                    bash /home/lizz_lab/cse12232433/project/colmena/multisite_/finetuning-surrogates/runs/analysis/monitor.sh &
                    ''',
                    walltime="01:00:00",
                ),
            ),
            HighThroughputExecutor(
                address=address_by_hostname(),
                label="gpu",
                available_accelerators=4,
                max_workers=1,
                cores_per_worker=1,
                worker_port_range=(20000,30000),
                worker_logdir_root='/home/lizz_lab/cse12232433/parsl-logs',
                provider=SlurmProvider(
                    partition = "gpulab01",
                    channel = LocalChannel(
                        envs={},
                        script_dir=None,
                        userhome='/home/lizz_lab/cse12232433/project/colmena/multisite_/finetuning-surrogates/',
                    ),
                    nodes_per_block=1,
                    init_blocks=1,
                    worker_init = '''
                    source /home/lizz_lab/cse12232433/miniconda3/bin/activate /home/lizz_lab/cse12232433/miniconda3/envs/multisite
                    bash /home/lizz_lab/cse12232433/project/colmena/multisite_/finetuning-surrogates/runs/analysis/monitor.sh &
                    ''',
                    walltime="01:00:00",
                ),
            )]
    )
        
    return config

def csecluster_RT_scale(log_dir: str) -> Config:
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
                worker_logdir_root='/home/lizz_lab/cse12232433/parsl-logs',
                provider=LocalProvider(
                    # min_blocks=1,
                    # max_blocks=4,
                    init_blocks=4,
                    # channels=[SSHChannel(hostname='gpu001',port='22', username='cse12232433', password='Yxx!199871!', script_dir='/home/lizz_lab/cse12232433/parsl-logs')],
                    channel=LocalChannel(),
                    worker_init='''
                    # Activate conda environment
                    source /home/lizz_lab/cse12232433/.bashrc
                    source /home/lizz_lab/cse12232433/miniconda3/bin/activate /home/lizz_lab/cse12232433/miniconda3/envs/multisite
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
                worker_logdir_root='/home/lizz_lab/cse12232433/parsl-logs',
                provider=LocalProvider(
                    # min_blocks=1,
                    # max_blocks=4,
                    init_blocks=4,
                    # channels=[SSHChannel(hostname='gpu001',port='22', username='cse12232433', password='Yxx!199871!', script_dir='/home/lizz_lab/cse12232433/parsl-logs')],
                    channel=LocalChannel(),
                    worker_init='''
                    # Activate conda environment
                    source /home/lizz_lab/cse12232433/.bashrc
                    source /home/lizz_lab/cse12232433/miniconda3/bin/activate /home/lizz_lab/cse12232433/miniconda3/envs/multisite
                    # bash /home/lizz_lab/cse12232433/project/colmena/multisite_/finetuning-surrogates/runs/analysis/monitor.sh &
                    # export TF_GPU_ALLOCATOR=cuda_malloc_async
                    which python
                    ''',
                ),
            )]
    )
        
    return config

def csecluster_RT_workqueue(log_dir: str) -> Config:
    config = Config(
        run_dir=log_dir,
        strategy='none',
        retries=1,
        executors=[
            WorkQueueExecutor(
                label="cpu",
                shared_fs=True,
                address=address_by_hostname(),
                provider=LocalProvider(
                # min_blocks=1,
                # max_blocks=4,
                init_blocks=4,
                # channels=[SSHChannel(hostname='gpu001',port='22', username='cse12232433', password='Yxx!199871!', script_dir='/home/lizz_lab/cse12232433/parsl-logs')],
                channel=LocalChannel(),
                worker_init='''
                source /home/lizz_lab/cse12232433/.bashrc
                source /home/lizz_lab/cse12232433/miniconda3/bin/activate /home/lizz_lab/cse12232433/miniconda3/envs/multisite
                which python
                ''',
                ),
            ),
            WorkQueueExecutor(
                label="gpu",
                shared_fs=True,
                address=address_by_hostname(),
                provider=LocalProvider(
                # min_blocks=1,
                # max_blocks=4,
                init_blocks=4,
                # channels=[SSHChannel(hostname='gpu001',port='22', username='cse12232433', password='Yxx!199871!', script_dir='/home/lizz_lab/cse12232433/parsl-logs')],
                channel=LocalChannel(),
                worker_init='''
                source /home/lizz_lab/cse12232433/.bashrc
                source /home/lizz_lab/cse12232433/miniconda3/bin/activate /home/lizz_lab/cse12232433/miniconda3/envs/multisite
                which python
                ''',
                ),
            ),
        ]
    )
        
    return config


def wsl(log_dir: str) -> Config:
    config = Config(
        run_dir=log_dir,
        retries=1,
        executors=[
            HighThroughputExecutor(
                label="cpu",
                max_workers=8,
                address=address_by_hostname(),
                worker_ports=(11001, 11002),  # Hard coded to match up with SSH tunnels
                worker_logdir_root='/home/yxx/parsl-logs',
                provider=AdHocProvider(
                    channels=[SSHChannel(hostname='127.0.0.1',port='10022', username='yxx', password='630824252', script_dir='/home/yxx/parsl-logs')],
                    worker_init='''
                    # Activate conda environment
                    source /home/yxx/miniconda3/bin/activate /home/yxx/miniconda3/envs/multisite
                    which python
                    ''',
                ),
            ),
            HighThroughputExecutor(
                address='localhost',
                label="gpu",
                available_accelerators=1,
                worker_ports=(54928, 54875),  # Hard coded to match up with SSH tunnels
                worker_logdir_root='/home/yxx/parsl-logs',
                provider=AdHocProvider(
                    channels=[SSHChannel(hostname='127.0.0.1', port='10022', username='yxx', password='630824252', script_dir='/home/yxx/parsl-logs')],
                    worker_init='''
                    # Activate conda environment
                    source /home/yxx/miniconda3/bin/activate /home/yxx/miniconda3/envs/multisite
                    which python
                    ''',
                ),
            )]
    )
        
    return config

def wsl_local(log_dir: str) -> Config:
    """Configuration where simulation tasks run on Theta and ML tasks run on Lambda.

    Args:
        log_dir: Path to store monitoring DB and parsl logs
    Returns:
        (Config) Parsl configuration
    """
    config = Config(
        run_dir=log_dir,
        retries=1,
        executors=[
            HighThroughputExecutor(
                label="cpu",
                max_workers=8,
                address=address_by_hostname(),
                worker_ports=(11001, 11002),  # Hard coded to match up with SSH tunnels
                worker_logdir_root='/home/yxx/parsl-logs',
                provider=LocalProvider(
                    # script_dir='/home/yxx/parsl-logs',
                    worker_init='''
                    # Activate conda environment
                    source /home/yxx/miniconda3/bin/activate /home/yxx/miniconda3/envs/multisite
                    which python
                    ''',
                    ),
            ),
            HighThroughputExecutor(
                address='localhost',
                label="gpu",
                available_accelerators=1,
                worker_ports=(54928, 54875),  # Hard coded to match up with SSH tunnels
                worker_logdir_root='/home/yxx/parsl-logs',
                    provider=LocalProvider(
                    nodes_per_block=1,
                    init_blocks=1,
                    min_blocks=1,
                    max_blocks=1,
                    # script_dir='/home/yxx/parsl-logs',
                    worker_init='''
                    # Activate conda environment
                    source /home/yxx/miniconda3/bin/activate /home/yxx/miniconda3/envs/multisite
                    which python
                    ''',
                    ),
            )]
    )
        
    return config