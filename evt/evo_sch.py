# import modules
from ctypes import Union
from math import sin
from pathlib import Path
import logging
import shutil
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, ClassVar, Union, List
from copy import deepcopy
import json
from functools import partial, update_wrapper
import numpy as np
import time
import pickle
import random
from random import shuffle, sample

import sys

sys.path.append(r"/home/yxx/work/project/colmena/multisite_/my_util")
from data_structure import SingletonClass, SimulationTask, my_SimulationTask

# evo stargety here
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

@dataclass
class available_task(SingletonClass):
    # task_names: list[str] = field(default_factory=list)
    # task_ids: list[dict[str, int]] = field(default_factory=list)
    def __init__(self,task_names: set, task_ids: dict[str, int], task_datas=None):
        self.task_names = task_names
        self.task_ids = task_ids
        
    def add_task_id(self, task_name:str, task_id:Union[int, list[int]]):
        self.task_names.add(task_name)
        for i in task_id:
            self.task_ids[task_name].append(i)
    
    def remove_task_id(self, task_name, task_id):
        ## judge if there is task id
        if task_id not in self.task_ids[task_name]:
            print(f"task id {task_id} not in task name {task_name}")
            # logging.warning(f"task id {task_id} not in task name {task_name}")
            return
        self.task_ids[task_name].remove(task_id)
        if len(self.task_ids[task_name]) == 0:
            self.task_names.remove(task_name)
            
    def get_available_task_id(self, task_name):
        return self.task_ids.get(task_name)
    
    def get_all(self):
        return self.task_names, self.task_ids


def get_tasks_nums(total_tasks_nums: Dict[str, int] = None):
    # bundle task
    # TODO adjust task to be processed
    # if task nums is less, provision more resources
    # if task nums is more, provision less resources
    # aim to choose a batch of tasks could get max throughput
    # tasks_nums = min(remain_task, epoch_task * 2)
    # resources per task = max(1,resources * 2 / task_nums)
    # task nums = resources / resources per task

    tasks_nums = {"train": 0, "sampling": 0, "simulation": 8, "inference": 0}
    return tasks_nums


def get_resources():
    return {"cpu": 64, "gpu": 4}


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
    
def get_available_task_id(task_name):
    # TODO get available task id from hpc
    pass



def estimate_simulation_time(task):
    # temp test
    molecule_length = task_queue_audit[task['task_id']].atoms.get_positions().shape[0]
    cpu_cores = 1
    
    
    closest_length = min(length_times, key=lambda x: abs(x[0]-molecule_length))
    length_time = closest_length[1]

    closest_cores = min(core_times.keys(), key=lambda x: abs(x-cpu_cores))
    core_time = core_times[closest_cores]

    return length_time*core_time/70

def get_task_running_time(task):
    # TODO get task running time from hpc
    pass


def calculate_resource_usage(task_allocation):
    # used in heterogenous resources
    total_cpu_usage, total_gpu_usage = 0, 0
    for task, resources in task_allocation.items():
        total_cpu_usage += resources['cpu']
        total_gpu_usage += resources['gpu']

    return total_cpu_usage, total_gpu_usage


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
            simulation_time = estimate_simulation_time(task)
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


def dummy_allocate(ind: individual):

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
            total_time += estimate_simulation_time(task)
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
            weight = estimate_simulation_time(task) / total_time
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


def generate_population(population_size: int, tasks_nums: Dict[str, int], total_resources: dict):
    # generate population
    assert sum(tasks_nums.values()
               ) <= total_resources['cpu'], "Not enough CPUs for all tasks"
    population = []
    for id in range(population_size):

        ind = individual(tasks_nums=tasks_nums,
                         total_resources=total_resources)
        
        
        # if available_task.get_instance() is None:
        #     ## throw error, avail_task is not initialized
        #     raise ValueError("avail_task is not initialized")
        at = available_task()
        avail_train_task_id:list = at.get_available_task_id('train')
        avail_simulation_task_id:list = at.get_available_task_id('simulation')
        avail_sampling_task_id:list = at.get_available_task_id('sampling')
        avail_inference_task_id:list = at.get_available_task_id('inference')
        

        task_queue = []
        if(avail_train_task_id is not None):
            avail_id = random.sample(avail_train_task_id, ind.tasks_nums['train'])
            task_queue.extend([{'name': 'train',
                                "task_id": avail_id[i]}
                            for i in range(ind.tasks_nums['train'])])
            
        if(avail_sampling_task_id is not None):
            avail_id = random.sample(
                avail_sampling_task_id, ind.tasks_nums['sampling'])
            task_queue.extend([{'name': 'sampling',
                                "task_id": avail_id[i]}
                            for i in range(ind.tasks_nums['sampling'])])
        if(avail_simulation_task_id is not None):
            avail_id = random.sample(
                avail_simulation_task_id, ind.tasks_nums['simulation'])
            task_queue.extend([{'name': 'simulation',
                                "task_id": avail_id[i]}
                            for i in range(ind.tasks_nums['simulation'])])
        if(avail_inference_task_id is not None):
            avail_id = random.sample(
                avail_inference_task_id, ind.tasks_nums['inference'])
            task_queue.extend([{'name': 'inference',
                                "task_id": avail_id[i]}
                            for i in range(ind.tasks_nums['inference'])])

        ind.task_allocation = [{"name": task["name"], "task_id": task["task_id"], "resources": {
            "cpu": 1}} for task in task_queue]

        ind = dummy_allocate(ind)
        population.append(ind)

    return population

def random_add_task(ind):
    # if available_task.get_instance() is None:
    #     ## throw error, avail_task is not initialized
    #     raise ValueError("avail_task is not initialized")
    at = available_task()
    avail_train_task_id = at.get_available_task_id('train')
    avail_simulation_task_id = at.get_available_task_id('simulation')
    avail_sampling_task_id = at.get_available_task_id('sampling')
    avail_inference_task_id = at.get_available_task_id('inference')
    
    # if ind.tasks_nums['simulation'] >= simulation_totals:
    #     return
    if len(avail_simulation_task_id) <= ind.tasks_nums['simulation']:
        return
    if sum(ind.tasks_nums.values()) >= total_cpu:
        return
    
    ##TODO only consider simulation task here, need change
    
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
            # if next_population[i].tasks_nums['simulation'] >= len(avail_simulation_task_id):
            #     continue
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
    # scores = [fitness(ind) for ind in population]
    # population = [population[i] for i in np.argsort(scores)]
    return max(population, key=fitness)
    # return population[0]
    
    
## test evo_sch
if __name__ == '__main__':
    ## temp test
    out_dir = Path('/home/yxx/work/project/colmena/multisite_/my_test/ga_simulation_test')
    with open(out_dir / 'task_queue_audit.pkl', 'rb') as f:
        task_queue_audit = pickle.load(f)
    with open(out_dir / 'length_time', 'rb') as fp:
        length_times = pickle.load(fp)
    with open(out_dir / 'cpu_time', 'rb') as fp:
        core_times = pickle.load(fp)
    total_cpu = 64

    trainning_time = 100
    sampling_time = 100
    inference_time = 100
    
    
    task_submit=[]
    at = available_task("simulation", {"simulation":[i for i in range(24)]})
    print(at.get_available_task_id('simulation'))
    best_individual = run_ga(100, 10)
    task_submit.extend(best_individual.task_allocation)
    print(task_submit)
    pass