from parsl.executors import HighThroughputExecutor
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
        retries=1,
        executors=[
            HighThroughputExecutor(
                label="cpu",
                max_workers=8,
                address=address_by_hostname(),
                worker_port_range=(20000,30000),
                worker_logdir_root='/home/lizz_lab/cse30019698/parsl-logs',
                provider=AdHocProvider(
                    channels=[SSHChannel(hostname='127.0.0.1',port='10022', username='cse30019698', password='Yxx!199871!', script_dir='/home/cse30019698/parsl-logs')],
                    worker_init='''
                    # Activate conda environment
                    source /home/lizz_lab/cse30019698/software/miniconda3/bin/activate /home/lizz_lab/cse30019698/software/miniconda3/envs/multisite
                    which python
                    ''',
                ),
            ),
            HighThroughputExecutor(
                address=address_by_hostname(),
                label="gpu",
                available_accelerators=4,
                worker_port_range=(20000,30000),
                max_workers=1,
                worker_logdir_root='/home/lizz_lab/cse30019698/parsl-logs',
                provider=AdHocProvider(
                    channels=[SSHChannel(hostname='rtxgpu005', port='10022', username='cse30019698', password='Yxx!199871!', script_dir='/home/cse30019698/parsl-logs')],
                    worker_init='''
                    # Activate conda environment
                    source /home/lizz_lab/cse30019698/software/miniconda3/bin/activate /home/lizz_lab/cse30019698/software/miniconda3/envs/multisite
                    which python
                    ''',
                ),
            )]
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
    # Set a Theta config for using the KNL nodes with a single worker per node
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
                # provider=AdHocProvider(
                #     channels=[SSHChannel(hostname='127.0.0.1',port='10022', username='yxx', password='630824252', script_dir='/home/yxx/parsl-logs')],
                #     worker_init='''
                #     # Activate conda environment
                #     conda activate multisite
                #     which python
                #     ''',
                # ),
                provider=LocalProvider(
                    script_dir='/home/yxx/parsl-logs',
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
                # provider=AdHocProvider(
                #     channels=[SSHChannel(hostname='127.0.0.1', port='10022', username='yxx', password='630824252', script_dir='/home/yxx/parsl-logs')],
                #     worker_init='''
                #     # Activate conda environment
                #     conda activate multisite
                #     which python
                #     ''',
                # ),
                    provider=LocalProvider(
                    nodes_per_block=1,
                    init_blocks=1,
                    min_blocks=1,
                    max_blocks=1,
                    script_dir='/home/yxx/parsl-logs',
                    worker_init='''
                    # Activate conda environment
                    source /home/yxx/miniconda3/bin/activate /home/yxx/miniconda3/envs/multisite
                    which python
                    ''',
                    ),
            )]
    )
        
    return config