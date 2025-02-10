from parsl.executors import HighThroughputExecutor, WorkQueueExecutor
from parsl.providers import CobaltProvider, AdHocProvider, SlurmProvider, LocalProvider
from parsl.addresses import address_by_hostname
from parsl.launchers import AprunLauncher, SrunLauncher
from parsl.channels import SSHChannel, LocalChannel, SSHInteractiveLoginChannel
from parsl import Config
import os
import configparser


def create_executor_from_config(config_file: str, log_dir: str) -> (Config, dict):
    """
    Create a Parsl executor based on the provided configuration file.

    Args:
        config_file (str): Path to the configuration file.
        log_dir (str): Path to the directory where logs will be stored.

    Returns:
        Config: Parsl configuration object.
    """

    config = configparser.ConfigParser()
    config.read(config_file)

    # Get environment settings from config
    env_config = config['Environment']
    bashrc_path = env_config.get('bashrc_path')
    conda_path = env_config.get('conda_path') 
    conda_env = env_config.get('conda_env')
    user_path = os.path.expanduser('~')

    # worker_init = f'''
    #     # Activate conda environment
    #     source {bashrc_path}
    #     source {conda_path}/bin/activate {conda_env}
    #     which python
    # '''
        
    executors = []
    node_resources = {}
    for section in config.sections():
        if section == 'Environment':
            continue
        hostname = config.get(section, 'name')
        cpus = int(config.get(section, 'cpus'))
        gpus = int(config.get(section, 'gpus'))
        # 如果包含gpu_devices字段
        if config.has_option(section, 'gpu_devices'):
            gpu_devices = config.get(section, 'gpu_devices')

        if gpus > 0:
            executor = HighThroughputExecutor(
                label=hostname,
                max_workers=cpus,
                # available_accelerators=gpus,
                address=address_by_hostname(),
                worker_port_range=(20000, 30000),
                worker_logdir_root=log_dir + '/parsl-logs',
                provider=LocalProvider(
                    init_blocks=1,
                    channel=SSHChannel(
                            hostname=hostname,
                            port='22',
                            # username='cse12232433',
                            # password='Yxx!199871!',
                            script_dir=log_dir + '/parsl-logs'
                        ),
                    worker_init=f'''
                        # Activate conda environment
                        module load cuda/12.1
                        source {bashrc_path}
                        source {conda_path}/bin/activate {conda_env}
                        export PSI_SCRATCH={user_path}/scratch
                        export CUDA_VISIBLE_DEVICES={gpu_devices}
                        which python
                        '''
                )
            )
        else:
            executor = HighThroughputExecutor(
                label=hostname,
                max_workers=cpus,
                address=address_by_hostname(),
                worker_port_range=(20000, 30000),
                worker_logdir_root=log_dir + '/parsl-logs',
                provider=LocalProvider(
                    init_blocks=1,
                    channel=SSHChannel(
                            hostname=hostname,
                            port='22',
                            # username='cse12232433',
                            # password='Yxx!199871!',
                            script_dir=log_dir + '/parsl-logs'
                        ),
                    worker_init=f'''
                        # Activate conda environment
                        source {bashrc_path}
                        source {conda_path}/bin/activate {conda_env}
                        export PSI_SCRATCH={user_path}/scratch
                        which python
                        '''
                )
            )
        # executor = HighThroughputExecutor(
        #     label=hostname,
        #     max_workers=56,
        #     address=address_by_hostname(),
        #     worker_port_range=(20000,30000),
        #     worker_logdir_root='/home/lizz_lab/cse12232433/parsl-logs',
        #     provider=LocalProvider(
        #         # min_blocks=1,
        #         # max_blocks=4,
        #         init_blocks=1,
        #         # channels=[SSHChannel(hostname='gpu001',port='22', username='cse12232433', password='Yxx!199871!', script_dir='/home/lizz_lab/cse12232433/parsl-logs')],
        #         channel=SSHChannel(
        #                 hostname=hostname,
        #                 port='22',
        #                 username='cse12232433',
        #                 password='Yxx!199871!',
        #                 script_dir=log_dir + '/parsl-logs'
        #             ),
        #         worker_init='''
        #         # Activate conda environment
        #         source /home/lizz_lab/cse12232433/.bashrc
        #         source /home/lizz_lab/cse12232433/miniconda3/bin/activate /home/lizz_lab/cse12232433/miniconda3/envs/multisite
        #         # export CUDA_VISIBLE_DEVICES=0,1,2,3
        #         which python
        #         ''',
        #     ),
        # )

        executors.append(executor)
        node_resources[hostname] = {'cpu': cpus, 'gpu': gpus}

    return Config(
        run_dir=log_dir,
        strategy='none',
        retries=1,
        executors=executors
    ), node_resources
