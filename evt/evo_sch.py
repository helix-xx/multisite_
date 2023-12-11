# import modules
import copy
from ctypes import Union
from math import sin
from pathlib import Path
import logging
import re
import shutil
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from tarfile import LENGTH_LINK
from typing import Dict, Any, Optional, ClassVar, Union, List
from copy import deepcopy
import json
from functools import partial, update_wrapper
from networkx import could_be_isomorphic
import numpy as np
import time
import pickle
import random
from random import shuffle, sample
import concurrent.futures
import sys
sys.path.append(r"../")
from my_util.data_structure import *

# evo stargety here
@dataclass
class individual:
    # individual information
    # static variable, unique id for each individual
    _next_id: ClassVar[int] = 0
    individual_id: int = -1
    tasks_nums: Dict[str, int] = field(default_factory=dict)
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
    
    # hash
    def __hash__(self) -> int:
        sorted_allocation = sorted(self.task_allocation, key=lambda x: (x['name'], x['task_id']))
        return hash(str(sorted_allocation))

@dataclass
class available_task(SingletonClass):
    # task_names: list[str] = field(default_factory=list)
    # task_ids: list[dict[str, int]] = field(default_factory=list)
    # def __init__(self,task_names: set[str], task_ids: dict[str, int], task_datas=None):
    def __init__(self, task_ids: dict[str, list[int]]):
        # print(type(task_names))
        ## TODO task_names is set, but not work while get a str in it
        # self.task_names = set()
        # self.task_names = self.task_names.update(task_names)
        self.task_ids = task_ids
        
    def add_task_id(self, task_name:str, task_id:Union[int, list[int]]):
        # if task_name not in self.task_names:
        #     self.task_names.add(task_name)
        task_id = [task_id] if isinstance(task_id, int) else task_id
        for i in task_id:
            self.task_ids[task_name].append(i)
    
    def remove_task_id(self, task_name:str, task_id:Union[int, list[int]]):
        ## judge if there is task id
        task_id = [task_id] if isinstance(task_id, int) else task_id
        
        for i in task_id:
            if i not in self.task_ids[task_name]:
                print(f"task id {i} not in task name {task_name}")
                # logging.warning(f"task id {task_id} not in task name {task_name}")
                continue
            self.task_ids[task_name].remove(i)
            if len(self.task_ids[task_name]) == 0:
                self.task_ids.pop(task_name)
        # if len(self.task_ids[task_name]) == 0:
        #     # print type
        #     print(type(self.task_names))
        #     print(self.task_names)
        #     print(task_name)
        #     self.task_names.remove(task_name)
            
    def get_available_task_id(self, task_name):
        return self.task_ids.get(task_name)
    
    def get_all(self):
        return self.task_ids

class evosch:
    
    ## temp test
    # out_dir = Path('../my_test/ga_simulation_test')
    # with open(out_dir / 'task_queue_audit.pkl', 'rb') as f:
    #     task_queue_audit = pickle.load(f)
    #     task_queue_audit = task_queue_audit[:24]
    # with open(out_dir / 'length_time', 'rb') as fp:
    #     length_times = pickle.load(fp)
    # with open(out_dir / 'cpu_time', 'rb') as fp:
    #     cpu_times = pickle.load(fp)
        
    def __init__(self,task_queue_audit=None, length_times=None, cpu_times=None, total_cpu=None,at:available_task=None):
        self.his_population = set()
        self.task_queue_audit = task_queue_audit
        self.length_times = length_times
        self.core_times = cpu_times
        self.total_cpu = total_cpu
        self.at = at
        
    def get_tasks_nums(self,total_tasks_nums: Dict[str, int] = None):
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


    def get_resources(self):
        return {"cpu": 64, "gpu": 4}


    def get_piority(self,task):
        # TODO # get piority from workflow
        if task['name'] == 'train':
            return 1
        elif task['name'] == 'sampling':
            return 2
        elif task['name'] == 'simulation':
            return 3
        elif task['name'] == 'inference':
            return 4
    
    # def get_available_task_id(self, task_name):
    #     # TODO get available task id from hpc
    #     pass



    def estimate_simulation_time(self, task):        
        """estimate simulation time"""
        molecule_length = self.task_queue_audit[task['task_id']].atoms.get_positions().shape[0]
        cpu_cores = 1
        
        
        closest_length = min(self.length_times, key=lambda x: abs(x[0]-molecule_length))
        length_time = closest_length[1]

        closest_cores = min(self.core_times.keys(), key=lambda x: abs(x-cpu_cores))
        core_time = self.core_times[closest_cores]

        return length_time*core_time/40

    def get_task_running_time(self, task):
        # TODO get task running time from hpc
        pass


    # def calculate_resource_usage(self, task_allocation):
    #     # used in heterogenous resources
    #     total_cpu_usage, total_gpu_usage = 0, 0
    #     for task, resources in task_allocation.items():
    #         total_cpu_usage += resources['cpu']
    #         total_gpu_usage += resources['gpu']

    #     return total_cpu_usage, total_gpu_usage


    def evaluate_score(self, total_time, max_time):
        # total_time - (max_time * total_cpu)
        # maybe weight decide by utilization of resources
        weight = 0.1
        return total_time - (weight * (max_time * self.total_cpu))
        # weight = 1
        # return total_time - (weight * max_time)

    def easy_score(self, ind):
        """metric evaluate remain task schedule difficulty

        Args:
            ind (_type_): _description_
        """
        pass
    
    def get_max_total_time(self, ind):
        """get max total time and max time of this individual"""
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
                simulation_time = self.estimate_simulation_time(task)
                total_time += simulation_time
                max_time = max(max_time, simulation_time)
            if task['name'] == "inference":
                total_time += inference_time
                max_time = max(max_time, inference_time)
        
        return total_time, max_time
    
    def evaluate_remain_task(self, ind):
        """evaluate remain task schedule difficulty"""
        if sum(ind.tasks_nums.values()) >= sum([len(value) for value in self.at.get_all().values()]):
            return 0
        oringin_task_nums = ind.tasks_nums
        oringin_total_resources = ind.total_resources
        task_nums = {k:len(v) for k ,v in self.at.get_all().items()}
        task_nums = {key:task_nums[key] - oringin_task_nums[key]  for key in oringin_task_nums.keys()}
        
        m = sum(task_nums.values())/sum(oringin_task_nums.values())
        total_resources = {k:v*m for k,v in oringin_total_resources.items()}
        
        remain_ind = self.generate_population(1, task_nums, total_resources)[0]
        remain_total_time, remain_max_time = self.get_max_total_time(remain_ind)
        score = self.evaluate_score(remain_total_time, remain_max_time)
        return score/m
    
    def fitness(self,ind):
        # get total time as throughput
        # get max task time as smallest generation time
        total_time, max_time = self.get_max_total_time(ind)
        
        
        ind.total_time = total_time
        ind.max_time = max_time
        ind.score = self.evaluate_score(total_time, max_time)
        remain_score = self.evaluate_remain_task(ind)
        ind.score = 0.5*remain_score+0.5*ind.score
        return ind.score


    def dummy_allocate(self,ind: individual):

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
                total_time += self.estimate_simulation_time(task)
            if task['name'] == "inference":
                total_time += inference_time
        ind.total_time = total_time

        weights = {}
        temp_remaining_cpus = remaining_cpus
        for task in ind.task_allocation:
            if task['name'] == "train":
                weight = trainning_time / total_time
            elif task['name'] == "sampling":
                weight = sampling_time / total_time
            elif task['name'] == "simulation":
                weight = self.estimate_simulation_time(task) / total_time
            elif task['name'] == "inference":
                weight = inference_time / total_time

            weights[(task["name"], task["task_id"])] = weight
            extra_cpus = int(weight * temp_remaining_cpus)

            # for task_alloc in ind.task_allocation:
            #     if task_alloc["name"] == task["name"] and task_alloc["task_id"] == task["task_id"]:
            #         task_alloc["resources"]["cpu"] += extra_cpus
            #         remaining_cpus -= extra_cpus
            #         break
            task["resources"]["cpu"] += extra_cpus
            remaining_cpus -= extra_cpus

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

        # return ind


    def generate_population(self,population_size: int, tasks_nums: Dict[str, int], total_resources: dict):
        # generate population
        assert sum(tasks_nums.values()
                ) <= total_resources['cpu'], "Not enough CPUs for all tasks"
        population = []
        for id in range(population_size):

            ind = individual(tasks_nums=copy.deepcopy(tasks_nums),
                            total_resources=copy.deepcopy(total_resources))
            
            
            # if available_task.get_instance() is None:
            #     ## throw error, avail_task is not initialized
            #     raise ValueError("avail_task is not initialized")
            # at = available_task()
            
            avail_train_task_id:list = self.at.get_available_task_id('train')
            avail_simulation_task_id:list = self.at.get_available_task_id('simulation')
            avail_sampling_task_id:list = self.at.get_available_task_id('sampling')
            avail_inference_task_id:list = self.at.get_available_task_id('inference')
            

            task_queue = []
            ## TODO throw error if task nums is not enough
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

            self.dummy_allocate(ind)
            population.append(ind)

        return population

    def random_add_task(self, ind):
        # if available_task.get_instance() is None:
        #     ## throw error, avail_task is not initialized
        #     raise ValueError("avail_task is not initialized")
        # at = available_task()
        avail_train_task_id = self.at.get_available_task_id('train')
        avail_simulation_task_id = self.at.get_available_task_id('simulation')
        avail_sampling_task_id = self.at.get_available_task_id('sampling')
        avail_inference_task_id = self.at.get_available_task_id('inference')
        
        # if ind.tasks_nums['simulation'] >= simulation_totals:
        #     return
        if len(avail_simulation_task_id) <= ind.tasks_nums['simulation']:
            return
        if sum(ind.tasks_nums.values()) >= self.total_cpu:
            return
        
        ##TODO only consider simulation task here, need change
        filter_task_id = [d['task_id'] for d in ind.task_allocation if d['name'] == 'simulation']
        could_choice_task_id = [d for d in avail_simulation_task_id if d not in filter_task_id]
        # if len(could_choice_task_id) <=0 :
        #     return
        random_task = random.choice(could_choice_task_id)
        ind.task_allocation.append({"name": "simulation", "task_id": random_task, "resources": {"cpu": 1}})
        ind.tasks_nums['simulation'] += 1
        self.dummy_allocate(ind)

    def random_remove_task(self, ind):
        if sum(ind.tasks_nums.values()) <= 1:
            return
        if len(ind.task_allocation) <=0:
            return
        remove_task = random.choice(ind.task_allocation)
        ind.task_allocation.remove(remove_task)
        ind.tasks_nums[remove_task['name']] -= 1
        self.dummy_allocate(ind)
        
    def process_individual(self,individual):
        ## TODO for now we just keep same nums
        if sum(individual.tasks_nums.values()) >= sum([len(value) for value in self.at.get_all().values()]):
            return
        while(individual in self.his_population):
            self.random_add_task(individual)
            self.random_remove_task(individual)
        self.his_population.add(individual)
            
    def run_ga(self, pop_size, num_generations):
        # tasks_nums = {"train": 0, "sampling": 0, "simulation": 8, "inference": 0}
        tasks_nums = self.get_tasks_nums()  # get nums from workflow
        resources = self.get_resources()  # get resources from hpc
        population = self.generate_population(pop_size, tasks_nums, resources)
        self.his_population.update(population)
        for gen in range(num_generations):
            # keep size of population
            population = population[:pop_size]
            # use best half side instead worse half side
            population = population[pop_size // 2:] + [ind.copy() for ind in population[pop_size // 2:]]
            # choose the best half
            next_population = population[:pop_size // 2]
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for i in range(len(next_population)):
                    executor.submit(self.process_individual(next_population[i]))
            # for i in range(len(next_population)):
            #     print([ind.tasks_nums for ind in next_population])
            #     print([len(ind.task_allocation) for ind in next_population])
            #     self.process_individual(next_population[i])
                    
            scores = [self.fitness(ind) for ind in population]
            population = [population[i] for i in np.argsort(scores)]
            print(f"Generation {gen}: {population[-1].score}")
        return max(population, key=self.fitness)

    
    
## test evo_sch
if __name__ == '__main__':
    ## temp test
    out_dir = Path('../my_test/ga_simulation_test')
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
    # at = available_task("simulation", {"simulation":[i for i in range(24)]})
    ga = evosch(task_queue_audit, length_times, core_times,total_cpu=64,at=available_task({"simulation":[i for i in range(24)],"train":[],"sampling":[],"inference":[]}))
    print(ga.at.get_available_task_id("simulation"))
    while bool(sum([len(v) for v in ga.at.get_all().values()])):
        best_individual = ga.run_ga(100, 50)
        for task in best_individual.task_allocation:
            ga.at.remove_task_id(task['name'],task['task_id'])
        task_submit.extend(best_individual.task_allocation)
        print(ga.at.get_all())
        # print(len(task_submit))
    print(task_submit)
    ## sum cores
    # total_cpu = 0
    # for task in task_submit:
    #     total_cpu += task['resources']['cpu']
    # print(total_cpu)
    # print(sum(best_individual.tasks_nums.values()))
    pass