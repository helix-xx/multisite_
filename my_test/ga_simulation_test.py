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

import sys
sys.path.append('../../multisite_')
# sys.path.append('../my_util')
from evt.evo_sch import *
from my_util.data_structure import *



current_path = os.path.dirname(os.path.abspath(__file__))
out_dir = Path(current_path) / 'ga_simulation_test'
try:
    os.mkdir(out_dir)
    print("creat directory: " + str(out_dir))
except FileExistsError:
    print("directory already exists: " + str(out_dir))
    
    

    
# out_dir = Path('../my_test/ga_simulation_test')
out_dir = Path('../my_test/ga_simulation_test')
with open(out_dir / 'task_queue_audit.pkl', 'rb') as f:
    task_queue_audit = pickle.load(f)
    task_queue_audit = task_queue_audit[:24]
with open(out_dir / 'length_time', 'rb') as fp:
    length_times = pickle.load(fp)
with open(out_dir / 'cpu_time', 'rb') as fp:
    core_times = pickle.load(fp)
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



# task_submit=[]
# # at = available_task("simulation", {"simulation":[i for i in range(24)]})
# ga = evosch(task_queue_audit, length_times, core_times,total_cpu=64,at=available_task({"simulation":[i for i in range(24)],"train":[],"sampling":[],"inference":[]}))
# print(ga.at.get_available_task_id("simulation"))
# while bool(sum([len(v) for v in ga.at.get_all().values()])):
#     best_individual = ga.run_ga(100, 50)
#     for task in best_individual.task_allocation:
#         ga.at.remove_task_id(task['name'],task['task_id'])
#     task_submit.extend(best_individual.task_allocation)
#     print(ga.at.get_all())
#     # print(len(task_submit))
# print(task_submit)
task_submit=[{'name': 'simulation', 'task_id': 23, 'resources': {'cpu': 9}}, {'name': 'simulation', 'task_id': 12, 'resources': {'cpu': 9}}, {'name': 'simulation', 'task_id': 7, 'resources': {'cpu': 8}}, {'name': 'simulation', 'task_id': 2, 'resources': {'cpu': 9}}, {'name': 'simulation', 'task_id': 14, 'resources': {'cpu': 7}}, {'name': 'simulation', 'task_id': 6, 'resources': {'cpu': 6}}, {'name': 'simulation', 'task_id': 10, 'resources': {'cpu': 7}}, {'name': 'simulation', 'task_id': 9, 'resources': {'cpu': 9}}, {'name': 'simulation', 'task_id': 16, 'resources': {'cpu': 10}}, {'name': 'simulation', 'task_id': 11, 'resources': {'cpu': 10}}, {'name': 'simulation', 'task_id': 19, 'resources': {'cpu': 9}}, {'name': 'simulation', 'task_id': 5, 'resources': {'cpu': 6}}, {'name': 'simulation', 'task_id': 18, 'resources': {'cpu': 7}}, {'name': 'simulation', 'task_id': 1, 'resources': {'cpu': 7}}, {'name': 'simulation', 'task_id': 13, 'resources': {'cpu': 6}}, {'name': 'simulation', 'task_id': 17, 'resources': {'cpu': 9}}, {'name': 'simulation', 'task_id': 20, 'resources': {'cpu': 11}}, {'name': 'simulation', 'task_id': 15, 'resources': {'cpu': 4}}, {'name': 'simulation', 'task_id': 21, 'resources': {'cpu': 9}}, {'name': 'simulation', 'task_id': 0, 'resources': {'cpu': 11}}, {'name': 'simulation', 'task_id': 22, 'resources': {'cpu': 3}}, {'name': 'simulation', 'task_id': 8, 'resources': {'cpu': 8}}, {'name': 'simulation', 'task_id': 3, 'resources': {'cpu': 17}}, {'name': 'simulation', 'task_id': 4, 'resources': {'cpu': 1}}]


batch_start_time = time.time()
total_cpu_cores = 64
current_cpu_cores = 0
max_workers = 16

# Convert the task list into a queue
task_queue = deque(task_submit)
# task_batch = task_queue_audit[:24]
task_batch = [my_SimulationTask(simu_task=task_queue_audit[i]) for i in range(len(task_queue_audit))]
futures_map = {}
futures = []

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    while task_queue:
        task = task_queue.popleft()  # Remove the task from the queue
        cpu = 4
        # cpu = task['resources']['cpu']
        task_id = task['task_id']
        if cpu > total_cpu_cores - current_cpu_cores:
            # Not enough CPU cores available, put the task back to the queue
            task_queue.appendleft(task)
            # time.sleep(5)
            break
        # Enough CPU cores available, submit this task
        calc = dict(calc='psi4', method='pbe0-d3', basis='aug-cc-pvdz', num_threads=cpu)
        task_batch[task_id].start_time = time.time()
        future = executor.submit(_run_calculator, str(write_to_string(task_batch[task_id].simu_task.atoms, 'xyz')), calc, out_dir.as_posix())
        task_batch[task_id].temp_cores = cpu
        current_cpu_cores += cpu
        futures.append(future)
        futures_map[future] = task
        
        
    while futures:
        done_futures = []
        for future in concurrent.futures.as_completed(futures):
            done_futures.append(future)
            task = futures_map[future]
            value = future.result()
            task_id = task['task_id']
            atoms = read_from_string(value, 'json')
            running_time = time.time() - task_batch[task_id].start_time
            # task_batch[task_id].dft_time[task_batch[task_id].temp_cores] = running_time // error code
            task_batch[task_id].dft_time = {task_batch[task_id].temp_cores: running_time}
            # This task has completed, release its CPU cores
            current_cpu_cores -= task_batch[task_id].temp_cores
            # Check if there are any tasks that can be submitted now
            while task_queue:
                task = task_queue.popleft()
                # cpu = task['resources']['cpu']
                cpu = 4
                task_id = task['task_id']
                if cpu > total_cpu_cores - current_cpu_cores:
                    task_queue.appendleft(task)
                    break
                # Submit this task and update the CPU cores counter
                task_batch[task_id].start_time = time.time()
                calc = dict(calc='psi4', method='pbe0-d3', basis='aug-cc-pvdz', num_threads=cpu)
                future = executor.submit(_run_calculator, str(write_to_string(task_batch[task_id].simu_task.atoms, 'xyz')), calc, out_dir.as_posix())
                task_batch[task_id].temp_cores = cpu
                current_cpu_cores += cpu
                futures.append(future)
                futures_map[future] = task
        # Remove the done futures from the list
        for future in done_futures:
            futures.remove(future)

batch_time = time.time() - batch_start_time
print("batch time: " + str(batch_time))
with open(out_dir / 'task_queue_simulated_ga_4cores_per_task2', 'wb') as f:
    pickle.dump(task_batch, f)
