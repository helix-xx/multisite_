from pdb import run
from fff.simulation import run_calculator
from fff.simulation.utils import read_from_string, write_to_string
from _pytest.fixtures import fixture
import ase
from ase.build import molecule
from dataclasses import dataclass
from pathlib import Path
from typing  import Dict, Any, List, Optional
import os
import psutil
from functools import partial, update_wrapper
import json
import pickle
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
import scipy
# from memory_profiler import profile


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
    
current_path = os.path.dirname(os.path.abspath(__file__))
out_dir = Path(current_path) / 'non_linear_temp'
try:
    os.mkdir(out_dir)
    print("creat directory: " + str(out_dir))
except FileExistsError:
    print("directory already exists: " + str(out_dir))
# cpus = mp.cpu_count()
cpus = 16
calc = dict(calc='psi4', method='pbe0-d3', basis='aug-cc-pvdz', num_threads=cpus)

## load pickle file for sampling atoms
with open(out_dir / 'task_queue_audit', 'rb') as f:
    task_queue = pickle.load(f)

## test all simulation tasks for full cpu cores as prior
task_queue_simulated = []
for task in task_queue:
    atoms = task.atoms
    print(atoms)
    atoms.set_center_of_mass([0,0,0])
    xyz = write_to_string(atoms, 'xyz')
    start = time.time()
    value = run_calculator(xyz, calc=calc, temp_path=out_dir.as_posix())
    running_time = time.time() - start
    print("running time: " + str(running_time))
    atoms = read_from_string(value, 'json')
    task_queue_simulated.append(my_SimulationTask(simu_task=task, dft_energy=atoms.get_potential_energy(), dft_time={cpus:running_time}))
    
with open(out_dir / 'task_queue_simulated', 'wb') as f:
    pickle.dump(task_queue_simulated, f)
    

## test one task non-linear and predict other tasks
task = task_queue_simulated[0]
cpu_sets = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
for cpu in cpu_sets:
    atoms = task.simu_task.atoms
    print(atoms)
    atoms.set_center_of_mass([0,0,0])
    xyz = write_to_string(atoms, 'xyz')
    start = time.time()
    value = run_calculator(xyz, calc=calc, temp_path=out_dir.as_posix(), cpus=cpu)
    running_time = time.time() - start
    print("running time: " + str(running_time))
    atoms = read_from_string(value, 'json')
    # task.simu_task.dft_energy = atoms.get_potential_energy()
    task.simu_task.dft_time[cpu] = running_time
    
with open(out_dir / 'non_linear_task', 'wb') as f:
    pickle.dump(task, f)

## predict other tasks

## output arrangement for all tasks
