from unittest import result
from venv import logger
from fff.simulation.utils import write_to_string
from colmena.queue.base import ColmenaQueues
from colmena.queue.python import PipeQueues
from colmena.thinker import BaseThinker, agent, event_responder, result_processor, task_submitter
from colmena.task_server import ParslTaskServer
from colmena.models import Result
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from parsl.executors import HighThroughputExecutor, ThreadPoolExecutor
from parsl.providers import LocalProvider
from functools import partial, update_wrapper
from parsl.config import Config
from threading import Lock, Event
from datetime import datetime
from typing import List, Tuple, Union, Optional, Dict, Any
import numpy as np
import argparse
import logging
import json
import sys
import os

from dataclasses import dataclass
import ase
from typing import Optional, Dict, Any, List
import pickle
from fff.simulation import _run_calculator, run_calculator

from concurrent.futures import ProcessPoolExecutor
from ase.calculators.calculator import Calculator

@dataclass
class Trajectory:
    """Tracks the state of searching along individual trajectories

    We mark the starting point, the last point produced from sampling,
    and the last point we produced that has been validated
    """
    id: int  # ID number of the
    starting: ase.Atoms  # Starting point of the trajectory
    current_timestep = 0  # How many timesteps have been used so far
    last_validated: ase.Atoms = None  # Last validated point on the trajectory
    current: ase.Atoms = None  # Last point produced along the trajectory
    last_run_length: int = 0  # How long between current and last_validated
    name: str = None  # Name of the trajectory

    def __post_init__(self):
        self.last_validated = self.current = self.starting

    def update_current_structure(self, strc: ase.Atoms, run_length: int):
        """Update the structure that has yet to be updated

        Args:
            strc: Structure produced by sampling
            run_length: How many timesteps were performed in sampling run
        """
        self.current = strc.copy()
        self.last_run_length = run_length

    def set_validation(self, success: bool):
        """Set whether the trajectory was successfully validated

        Args:
            success: Whether the validation was successful
        """
        if success:
            self.last_validated = self.current  # Move the last validated forward
            self.current_timestep += self.last_run_length

@dataclass
class SimulationTask:
    atoms: ase.Atoms  # Structure to be run
    traj_id: int  # Which trajectory this came from
    ml_eng: float  # Energy predicted from machine learning model
    ml_std: Optional[float] = None  # Uncertainty of the model
    
@dataclass
class my_SimulationTask:
    simu_task: SimulationTask # basic information store in SimulationTask
    dft_energy: Optional[float] = None  # DFT energy of the structure
    dft_time: Optional[dict[int,float]] = None  # Dictionary to store DFT run times for different CPU cores
    temp_cores: Optional[int] = None  # Number of CPU cores used for the DFT calculation
    
CalcType = Union[Calculator, dict]
# def run_calculator(xyz: str, calc: CalcType, temp_path: Optional[str] = None, cpus:int = 1) -> str:
#     """Run an NWChem computation on the requested cluster

#     Args:
#         xyz: Cluster to evaluate
#         calc: ASE calculator to use. Either the calculator object or a dictionary describing the settings
#             (only Psi4 supported at the moment for dict)
#         temp_path: Base path for the scratch files
#     Returns:
#         Atoms after the calculation in a JSON format
#     """
#     if cpus > 1:
#         calc["num_threads"] = cpus
#     # Some calculators do not clean up their resources well
#     with ProcessPoolExecutor(max_workers=1) as exe:
#         fut = exe.submit(_run_calculator, str(xyz), calc, temp_path)  # str ensures proxies are resolved
#         return fut.result()
    
class Thinker(BaseThinker):
    def __init__(self, queues: ColmenaQueues, task_queue_audit, output_dir):
        super().__init__(queues)
        self.task_queue_audit = task_queue_audit
        self.output_dir = output_dir
    
    @task_submitter(task_type='simulate')
    def submit_simulattion(self):
        for task in self.task_queue_audit:
            atoms = task.atoms
            xyz = write_to_string(atoms, 'xyz')
            self.queues.send_inputs(xyz, method='run_calculator', topic='simulate',
                keep_inputs=True,  # The XYZ file is not big
                task_info={'traj_id': task.traj_id, 'task_type': "simulation",
                            'ml_energy': task.ml_eng, 'xyz': xyz})
    
    @result_processor(topic='simulate')
    def store_simulation(self, result: Result):
        with open(os.path.join(self.output_dir, 'simulation_results.json'), 'a') as f:
            print(result.json(),file=f)
        


if __name__ == '__main__':
    out_dir = os.path.join('colmena_test',f'-{datetime.now().strftime("%d%m%y-%H%M%S")}')
    os.makedirs(out_dir, exist_ok=False)
    # Set up the logging
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(out_dir, 'runtime.log')),
                                  logging.StreamHandler(sys.stdout)])
    num_parallel = os.cpu_count()
    
    with open('/home/yxx/work/project/colmena/multisite_/my_test/colmena_test/task_queue_audit', 'rb') as f:
        task_queue_audit = pickle.load(f)
    with open('/home/yxx/work/project/colmena/multisite_/my_test/ga_simulation_test/length_time', 'rb') as fp:
        length_times = pickle.load(fp)
    with open('/home/yxx/work/project/colmena/multisite_/my_test/ga_simulation_test/cpu_time', 'rb') as fp:
        core_times = pickle.load(fp)
    
    # Write the configuration
    config = Config(
        executors=[
            HighThroughputExecutor(
                address="localhost",
                label="simulation",
                max_workers=num_parallel,
                # cores_per_worker=0.0001, # default is 1
                worker_port_range=(10000, 20000),
                provider=LocalProvider(
                    init_blocks=1,
                    max_blocks=1,
                ),
            ),
            HighThroughputExecutor(
                address="localhost",
                label="task_generator",
                max_workers=1,
                provider=LocalProvider(
                    init_blocks=1,
                    max_blocks=1,
                ),
            ),
            ThreadPoolExecutor(label="local_threads", max_threads=4)
        ],
        strategy=None,
    )
    config.run_dir = os.path.join(out_dir, 'run-info')
    
    def _wrap(func, **kwargs):
        out = partial(func, **kwargs)
        update_wrapper(out, func)
        return out
    

    calc = dict(calc='psi4', method='pbe0-d3', basis='aug-cc-pvdz', num_threads=1)
    my_run_simulation = _wrap(run_calculator, calc = calc, temp_path=out_dir)
    methods = [(my_run_simulation, {'executors': ['simulation']})]
    
    def estimate_trainning_time(task, his=None, queue=None):
        return 10

    def estimate_sampling_time(task, his=None, queue=None):
        return 10

    def estimate_simulation_time(task, his=None, queue=None):
        molecule_length = int(queue.result_list[task['task_id']].inputs[0][0].split('\n')[0])
        cpu_cores = task['resources']['cpu']
        length_times = his['length_times']
        core_times = his['core_times']
        # logger.info(f"molecule_length: {molecule_length}, cpu_cores: {cpu_cores}")
        # logger.info(f"length_times: {length_times}, core_times: {core_times}")
        closest_length = min(length_times.keys(), key=lambda x: abs(x-molecule_length))
        length_time = length_times[closest_length]

        closest_cores = min(core_times.keys(), key=lambda x: abs(x-cpu_cores))
        core_time = core_times[closest_cores]

        return length_time*core_time/40

    def estimate_inference_time(task, his=None, queue=None):
        return 10
    # topics=['simulate', 'sample', 'train', 'infer']
    estimate_methods = {'train': estimate_trainning_time, 'sample': estimate_sampling_time, 'simulate': estimate_simulation_time, 'infer': estimate_inference_time}
    
    queues = PipeQueues(keep_inputs=True, topics=['simulate'],estimate_methods=estimate_methods)
    queues.evosch.hist_data.add_data('length_times', length_times)
    queues.evosch.hist_data.add_data('core_times', core_times)
    doer = ParslTaskServer(methods, queues, config)
    thinker = Thinker(queues, task_queue_audit, out_dir)
    
    try:
        logging.info("Starting the task server")
        doer.start()
        thinker.start()
        
        thinker.join()
        
    finally:
        queues.send_kill_signal()
    doer.join()
    