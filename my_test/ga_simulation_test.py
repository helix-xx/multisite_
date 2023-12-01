from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
from pdb import run
from fff.simulation import run_calculator,_run_calculator
from fff.simulation.utils import read_from_string, write_to_string
from _pytest.fixtures import fixture
import ase
from ase.build import molecule
from dataclasses import dataclass, field
from pathlib import Path
from typing  import Dict, Any, List, Optional, ClassVar
import os
import psutil
from functools import partial, update_wrapper
import json
import pickle
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
import scipy
import random
from copy import deepcopy
import numpy as np
from collections import deque
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
    temp_cores: Optional[int] = None  # Number of CPU cores used for the DFT calculation
    
    
    
    




current_path = os.path.dirname(os.path.abspath(__file__))
out_dir = Path(current_path) / 'ga_simulation_test'
try:
    os.mkdir(out_dir)
    print("creat directory: " + str(out_dir))
except FileExistsError:
    print("directory already exists: " + str(out_dir))
    
    
with open(out_dir / 'task_queue_audit.pkl', 'rb') as f:
    task_queue_audit = pickle.load(f)
    
task_queue_audit = task_queue_audit[:24]
    

# resources
total_cpu = 64

# a batch runs
train_nums = 4
sampling_nums = 20
simulation_nums = 20
inference_nums = 4

# time consume
trainning_time = 50
sampling_time = 10
inference_time = 10
# simulation time
# nonlinear performance for cpu cores
# nonlinear preformance for atoms length
with open(out_dir / 'length_time', 'rb') as fp:
    length_time = pickle.load(fp)
with open(out_dir / 'cpu_time', 'rb') as fp:
    cpu_time = pickle.load(fp)

excluded_simulation_numbers = set()
avail_simulation_task_id = [i for i in range(len(task_queue_audit))]
avail_train_task_id = [i for i in range(train_nums)]
avail_sampling_task_id = [i for i in range(sampling_nums)]
avail_inference_task_id = [i for i in range(inference_nums)]


# TODO make a data class for task in evolution algorithm
# make a individual a class or make a task a class?
@dataclass
class individual:
    # individual information
    # static variable, unique id for each individual
    _next_id: ClassVar[int] = 0
    individual_id: int = -1
    tasks_nums: Dict[str, int] = field(default_factory=dict)
    # run_train_task_id: list[int] = field(default_factory=list)
    # run_simulation_task_id: list[int] = field(default_factory=list)
    # run_sampling_task_id: list[int] = field(default_factory=list)
    # run_inference_task_id: list[int] = field(default_factory=list)
    total_resources: dict = field(default_factory=dict)
    total_time: int = 0
    max_time: int = 0
    score: int = 0

    # allocation information
    task_allocation: list[dict[str, int]] = field(default_factory=list)
    
    # initialize individual id
    def __post_init__(self):
        if self.individual_id == -1:
            self.individual_id = individual._next_id
            individual._next_id += 1

    # copy individual
    def copy(self):
        copied_individual = deepcopy(self)
        copied_individual.individual_id = individual._next_id
        individual._next_id += 1
        return copied_individual


def get_tasks_nums(total_tasks_nums: Dict[str, int] = None):

    tasks_nums = {"train": 0, "sampling": 0, "simulation": 8, "inference": 0}
    return tasks_nums


def get_resources():
    return {"cpu": 64, "gpu": 4}


def dummy_allocate(ind:individual):

    # TODO only cpu here, gpu is not considered
    total_cpus = ind.total_resources['cpu']
    remaining_cpus = total_cpus - sum(ind.tasks_nums.values())
    if remaining_cpus < 0:
        raise ValueError("Not enough CPUs for all tasks")
    # assert sum(ind.tasks_nums.values()) <= ind.total_resources['cpu'], "Not enough CPUs for all tasks"
    for task in ind.task_allocation:
        task["resources"]["cpu"] = 1

    total_time = 0
    for task in ind.task_allocation:
        if task['name'] == "train":
            total_time += trainning_time
        if task['name'] == "sampling":
            total_time += sampling_time
        if task['name'] == "simulation":
            total_time += estimate_simulation_time(
                task_queue_audit[task['task_id']].atoms.get_positions().shape[0], 1, length_time, cpu_time)
        if task['name'] == "inference":
            total_time += inference_time
    ind.total_time = total_time

    weights = {}
    for task in ind.task_allocation:
        if task['name'] == "train":
            weight = trainning_time / total_time
        elif task['name'] == "sampling":
            weight = sampling_time / total_time
        elif task['name'] == "simulation":
            weight = estimate_simulation_time(task_queue_audit[task['task_id']].atoms.get_positions(
            ).shape[0], 1, length_time, cpu_time) / total_time
        elif task['name'] == "inference":
            weight = inference_time / total_time

        weights[(task["name"], task["task_id"])] = weight
        extra_cpus = int(weight * remaining_cpus)

        for task_alloc in ind.task_allocation:
            if task_alloc["name"] == task["name"] and task_alloc["task_id"] == task["task_id"]:
                task_alloc["resources"]["cpu"] += extra_cpus
                remaining_cpus -= extra_cpus
                break

    # if there are still remaining CPUs due to rounding, assign them to tasks
    # based on their weights
    for task_key, _ in sorted(weights.items(), key=lambda item: item[1], reverse=True):
        if remaining_cpus <= 0:
            break
        for task_alloc in ind.task_allocation:
            if task_alloc["name"] == task_key[0] and task_alloc["task_id"] == task_key[1]:
                task_alloc["resources"]["cpu"] += 1
                remaining_cpus -= 1
                break

    return ind


def get_piority(task):
    # TODO # get piority from workflow
    if task['name'] == 'train':
        return 1
    elif task['name'] == 'sampling':
        return 2
    elif task['name'] == 'simulation':
        return 3
    elif task['name'] == 'inference':
        return 4


def estimate_simulation_time(molecule_length, cpu_cores, length_times, core_times):
    closest_length = min(length_times, key=lambda x: abs(x[0]-molecule_length))
    length_time = closest_length[1]

    closest_cores = min(core_times.keys(), key=lambda x: abs(x-cpu_cores))
    core_time = core_times[closest_cores]

    return length_time*core_time/70


def calculate_resource_usage(task_allocation):
    total_cpu_usage, total_gpu_usage = 0, 0
    for task, resources in task_allocation.items():
        total_cpu_usage += resources['cpu']
        total_gpu_usage += resources['gpu']

    return total_cpu_usage, total_gpu_usage


def generate_population(population_size: int, tasks_nums: Dict[str, int], total_resources: dict):
    # generate population
    assert sum(tasks_nums.values()) <= total_resources['cpu'], "Not enough CPUs for all tasks"
    population = []
    for id in range(population_size):
        
        ind = individual(tasks_nums=tasks_nums, total_resources=total_resources)
        global avail_train_task_id
        global avail_simulation_task_id
        global avail_sampling_task_id
        global avail_inference_task_id

        task_queue = []
        avail_id = random.sample(avail_train_task_id, ind.tasks_nums['train'])
        task_queue.extend([{'name': 'train',
                            "task_id": avail_id[i]}
                        for i in range(ind.tasks_nums['train'])])
        avail_id = random.sample(avail_sampling_task_id, ind.tasks_nums['sampling'])
        task_queue.extend([{'name': 'sampling',
                            "task_id": avail_id[i]}
                        for i in range(ind.tasks_nums['sampling'])])
        avail_id = random.sample(avail_simulation_task_id, ind.tasks_nums['simulation'])
        task_queue.extend([{'name': 'simulation',
                            "task_id": avail_id[i]}
                        for i in range(ind.tasks_nums['simulation'])])
        avail_id = random.sample(avail_inference_task_id, ind.tasks_nums['inference'])
        task_queue.extend([{'name': 'inference',
                            "task_id": avail_id[i]}
                        for i in range(ind.tasks_nums['inference'])])

        ind.task_allocation = [{"name": task["name"], "task_id": task["task_id"], "resources": {
            "cpu": 1}} for task in task_queue]
        
        ind = dummy_allocate(ind)
        population.append(ind)

    return population


def evaluate_score(total_time, max_time):
    # total_time - (max_time * total_cpu)
    # maybe weight decide by utilization of resources
    weight = 0.1
    return total_time - (weight * (max_time * total_cpu))


def fitness(ind):
    # get total time as throughput
    # get max task time as smallest generation time
    total_time = 0
    max_time = 0
    for task in ind.task_allocation:
        if task['name'] == "train":
            total_time += trainning_time
            # max_time = max(max_time, trainning_time)
        if task['name'] == "sampling":
            total_time += sampling_time
            # max_time = max(max_time, sampling_time)
        if task['name'] == "simulation":
            simulation_time = estimate_simulation_time(
                task_queue_audit[task['task_id']].atoms.get_positions().shape[0], 1, length_time, cpu_time)
            total_time += simulation_time
            max_time = max(max_time, simulation_time)
        if task['name'] == "inference":
            total_time += inference_time
            max_time = max(max_time, inference_time)
    ind.total_time = total_time
    ind.max_time = max_time
    ind.score = evaluate_score(total_time, max_time)
    # TODO weights of this two metric
    # How to measure the throughput that this batch of reduced time can increase?
    # sum positive total time and negative max time as score
    return ind.score



def random_add_task(ind):
    global avail_train_task_id
    global avail_simulation_task_id
    global avail_sampling_task_id
    global avail_inference_task_id
    if ind.tasks_nums['simulation'] >= len(avail_simulation_task_id):
        return
    if len(avail_simulation_task_id) <= ind.tasks_nums['simulation']:
        return
    if sum(ind.tasks_nums.values()) >= total_cpu:
        return
    
    ##TODO only consider simulation task here
    filter_task_id = [d['task_id'] for d in ind.task_allocation if d['name'] == 'simulation']
    could_choice_task_id = [d for d in avail_simulation_task_id if d not in filter_task_id]
    random_task = random.choice(could_choice_task_id)
    ind.task_allocation.append({"name": "simulation", "task_id": random_task, "resources": {"cpu": 1}})
    ind.tasks_nums['simulation'] += 1
    dummy_allocate(ind)

def random_remove_task(ind):
    if sum(ind.tasks_nums.values()) <= 1:
        return
    remove_task = random.choice(ind.task_allocation)
    ind.task_allocation.remove(remove_task)
    ind.tasks_nums[remove_task['name']] -= 1
    dummy_allocate(ind)


def run_ga(pop_size, num_generations):
    # tasks_nums = {"train": 0, "sampling": 0, "simulation": 8, "inference": 0}
    tasks_nums = get_tasks_nums()  # get nums from workflow
    resources = get_resources()  # get resources from hpc
    population = generate_population(pop_size, tasks_nums, resources)

    for gen in range(num_generations):
        # keep the size
        population = population[:pop_size]
        # use best half side instead worse half side
        population = population[pop_size // 2:] + [ind.copy() for ind in population[pop_size // 2:]]
        # choose the best half
        next_population = population[:pop_size // 2]
        for i in range(pop_size//2):
            ## TODO for now we just keep same nums
            if next_population[i].tasks_nums['simulation'] >= len(avail_simulation_task_id):
                continue
            # random add
            random_add_task(next_population[i])
            # random remove
            random_remove_task(next_population[i])
        # population.extend(next_population)
        # print individual id
        # iid = [i.individual_id for i in population]
        # print(iid)
        # print(len(iid))
        scores = [fitness(ind) for ind in population]
        population = [population[i] for i in np.argsort(scores)]
        print(f"Generation {gen}: {population[-1].score}")
    return max(population, key=fitness)

task_submit=[]
print(avail_simulation_task_id)
best_individual = run_ga(100, 10)
task_submit.extend(best_individual.task_allocation)
print(best_individual)
filter_task_id = [d['task_id'] for d in best_individual.task_allocation if d['name'] == 'simulation']
avail_simulation_task_id = [d for d in avail_simulation_task_id if d not in filter_task_id]
print(avail_simulation_task_id)
best_individual = run_ga(100, 10)
task_submit.extend(best_individual.task_allocation)
print(best_individual)
filter_task_id = [d['task_id'] for d in best_individual.task_allocation if d['name'] == 'simulation']
avail_simulation_task_id = [d for d in avail_simulation_task_id if d not in filter_task_id]
print(avail_simulation_task_id)
best_individual = run_ga(100, 10)
task_submit.extend(best_individual.task_allocation)
print(best_individual)
filter_task_id = [d['task_id'] for d in best_individual.task_allocation if d['name'] == 'simulation']
avail_simulation_task_id = [d for d in avail_simulation_task_id if d not in filter_task_id]
print(avail_simulation_task_id)
print(task_submit)




batch_start_time = time.time()
total_cpu_cores = 64
current_cpu_cores = 0
max_workers = 16

# Convert the task list into a queue
task_queue = deque(task_submit)
task_batch = task_queue_audit[:24]
start_time =[]


with ProcessPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    while task_queue:
        task = task_queue.popleft()  # Remove the task from the queue
        cpu = task['resources']['cpu']
        task_id = task['task_id']
        if cpu > total_cpu_cores - current_cpu_cores:
            # Not enough CPU cores available, put the task back to the queue
            task_queue.appendleft(task)
            # time.sleep(5)
            break
        # Enough CPU cores available, submit this task
        calc = dict(calc='psi4', method='pbe0-d3', basis='aug-cc-pvdz', num_threads=cpu)
        start_time.append(time.time())
        future = executor.submit(_run_calculator, str(write_to_string(task_batch[task_id], 'xyz')), calc, out_dir.as_posix())
        task.temp_cores = cpu
        current_cpu_cores += cpu
        futures.append((task, future))
    for task, future in concurrent.futures.as_completed(futures):
        value = future.result()
        atoms = read_from_string(value, 'json')
        running_time = time.time() - batch_start_time
        task.dft_time[task.temp_cores] = running_time
        # This task has completed, release its CPU cores
        current_cpu_cores -= task.temp_cores
        # Check if there are any tasks that can be submitted now
        while task_queue:
            task = task_queue.popleft()
            cpu = task['resources']['cpu']
            task_id = task['task_id']
            if cpu > total_cpu_cores - current_cpu_cores:
                task_queue.append(task)
                break
            # Submit this task and update the CPU cores counter
            future = executor.submit(_run_calculator, str(write_to_string(task_batch[task_id], 'xyz')), calc, out_dir.as_posix())
            task.temp_cores = cpu
            current_cpu_cores += cpu
            futures.append((task, future))

batch_time = time.time() - batch_start_time
print("batch time: " + str(batch_time))
with open(out_dir / 'task_queue_simulated_assign_parrallel', 'wb') as f:
    pickle.dump(task_batch, f)