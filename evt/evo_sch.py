# import modules
from asyncio import futures
import copy
from ctypes import Union
from itertools import accumulate
from math import e, sin
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
from anyio import current_time
from matplotlib.style import available
from moldesign import score
from networkx import could_be_isomorphic
import numpy as np
import time
import pickle
import random
from random import randint, shuffle, sample
import concurrent.futures
import sys
import heapq

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
    
    ## optional
    predict_run_seq: list = field(default_factory=list)

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
    
    def get_resources(self, task_name, task_id):
        for task in self.task_allocation:
            if task['name'] == task_name and task['task_id'] == task_id:
                return task['resources']
        return None

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
    
    def get_task_nums(self):
        result = {}
        for key,value in self.task_ids.items():
            result[key] = len(value)
        return result

class evosch:
        
    def __init__(self,task_queue_audit=None, length_times=None, cpu_times=None, total_cpu=None,at:available_task=None):
        self.his_population = set()
        self.task_queue_audit = task_queue_audit
        self.length_times = length_times
        self.core_times = cpu_times
        self.total_cpu = total_cpu
        self.at = at
        
    def get_batch_tasks_nums(self,total_tasks_nums: Dict[str, int] = None):
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
        tasks_nums = self.get_batch_tasks_nums()  # get nums from workflow
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
    
class evosch2:
    """add all task in individual
    cost function calculate total time idle time
    """
    def __init__(self,task_queue_audit=None, length_times=None, cpu_times=None, resources=None,at:available_task=None):
        self.his_population = set()
        self.task_queue_audit = task_queue_audit
        self.length_times = length_times
        self.core_times = cpu_times
        self.resources:dict = resources
        self.at = at # available task
    
    def get_task_nums(self):
        pass
    def get_resources(self):
        return self.resources
    def get_piority(self):
        pass
    
    def estimate_simulation_time(self, task, cpu_cores=1):        
        molecule_length = self.task_queue_audit[task['task_id']].atoms.get_positions().shape[0]
        cpu_cores = cpu_cores
        
        
        closest_length = min(self.length_times, key=lambda x: abs(x[0]-molecule_length))
        length_time = closest_length[1]

        closest_cores = min(self.core_times.keys(), key=lambda x: abs(x-cpu_cores))
        core_time = self.core_times[closest_cores]

        return length_time*core_time/40
    
    def calculate_completion_time(self, ind:individual):
        available_cpu = ind.total_resources['cpu']
        current_time = 0
        ongoing_task = []
        for task in ind.task_allocation:
            while ongoing_task and ongoing_task[0][0] <= current_time:
                _, cpus = heapq.heappop(ongoing_task)
                available_cpu += cpus
                
            while available_cpu < task['resources']['cpu']:
                if ongoing_task:
                    next_finish_time, cpus = heapq.heappop(ongoing_task)
                    current_time = next_finish_time
                    available_cpu += cpus
                else:
                    break
            
            if available_cpu < task['resources']['cpu']:
                raise ValueError("Not enough CPUs for all tasks")
            
            available_cpu -= task['resources']['cpu']
            finish_time = current_time + self.estimate_simulation_time(task,task['resources']['cpu'])
            heapq.heappush(ongoing_task, (finish_time, task['resources']['cpu']))
        while ongoing_task:
            next_finish_time, cpus = heapq.heappop(ongoing_task)
            available_cpu += cpus
            current_time = next_finish_time
        return current_time
    
    def calculate_completion_time_record(self, ind):
        available_cpu = ind.total_resources['cpu']
        current_time = 0
        ongoing_task = []
        running_seq = []  # 记录任务执行的顺序和时间

        for task in ind.task_allocation:
            # 检查是否有任务已经完成
            while ongoing_task and ongoing_task[0][0] <= current_time:
                _, cpus, finished_task_id, task_name = heapq.heappop(ongoing_task)
                available_cpu += cpus
                # 更新任务的完成时间
                for task_record in running_seq:
                    if task_record['task_id'] == finished_task_id and task_record['name'] == task_name:
                        task_record['finish_time'] = current_time
                        task_record['total_runtime'] = current_time - task_record['start_time']
                        break

            # 等待直到有足够的CPU资源
            while available_cpu < task['resources']['cpu']:
                if ongoing_task:
                    next_finish_time, cpus, finished_task_id, task_name = heapq.heappop(ongoing_task)
                    for task_record in running_seq:
                        if task_record['task_id'] == finished_task_id and task_record['name'] == task_name:
                            task_record['finish_time'] = current_time
                            task_record['total_runtime'] = current_time - task_record['start_time']
                            break
                    current_time = next_finish_time
                    available_cpu += cpus
                else:
                    break

            if available_cpu < task['resources']['cpu']:
                raise ValueError("Not enough CPUs for all tasks")

            # 开始新任务
            available_cpu -= task['resources']['cpu']
            start_time = current_time
            finish_time = current_time + self.estimate_simulation_time(task, task['resources']['cpu'])
            heapq.heappush(ongoing_task, (finish_time, task['resources']['cpu'], task['task_id'], task['name']))

            # 记录任务的开始时间和其他信息
            running_seq.append({
                'name': task['name'],
                'task_id': task['task_id'],
                'start_time': start_time,
                'finish_time': None,  # 将在任务完成时更新
                'total_runtime': None  # 将在任务完成时更新
            })

        # 清空剩余的任务并记录完成时间
        while ongoing_task:
            next_finish_time, cpus, finished_task_id, task_name = heapq.heappop(ongoing_task)
            available_cpu += cpus
            current_time = next_finish_time
            # 更新任务的完成时间
            for task_record in running_seq:
                if task_record['task_id'] == finished_task_id and task_record['name'] == task_name:
                    task_record['finish_time'] = current_time
                    task_record['total_runtime'] = current_time - task_record['start_time']
                    break
        
        ind.predict_run_seq = running_seq
        # 返回总完成时间和任务运行序列
        return current_time
    
    def calculate_total_time(self, ind:individual):
        total_time = 0
        for task in ind.task_allocation:
            total_time += self.estimate_simulation_time(task,task['resources']['cpu'])
        return total_time
            
    
    
    def fitness(self, ind:individual):
        # calculate total time based on avail resources and task
        total_time = 0  ## time accumulate by all task
        completion_time = 0 ## HPC makespan

        total_time = self.calculate_total_time(ind)
        completion_time = self.calculate_completion_time_record(ind)
        
        # ind.score = 1000/completion_time
        ind.score = -completion_time
        return ind.score
        
    
    def initialize_individual(self):
        pass
    
    def generate_population(self, population_size: int):
        ## add all task to individual
        task_nums = self.at.get_task_nums()
        all_tasks = self.at.get_all()
        population = []
        for _ in range(population_size):
            ind = individual(tasks_nums=copy.deepcopy(task_nums),total_resources=copy.deepcopy(self.get_resources()))
            
            task_queue = []
            for name, ids in all_tasks.items():
                for task_id in ids:
                    new_task = {
                        "name":name,
                        "task_id": task_id,
                        "resources":{
                            "cpu": random.randint(1,16)
                            # "cpu": 1
                        }
                    }
                    task_queue.append(new_task)
            random.shuffle(task_queue)
        
            ind.task_allocation = task_queue
            population.append(ind)
            
        for _ in range(population_size):
            ind = individual(tasks_nums=copy.deepcopy(task_nums),total_resources=copy.deepcopy(self.get_resources()))
            
            task_queue = []
            for name, ids in all_tasks.items():
                for task_id in ids:
                    new_task = {
                        "name":name,
                        "task_id": task_id,
                        "resources":{
                            # "cpu": random.randint(1,16)
                            "cpu": 1
                        }
                    }
                    task_queue.append(new_task)
            random.shuffle(task_queue)
        
            ind.task_allocation = task_queue
            population.append(ind)
            
        return population 
    
    def mutate_cpu(self,ind:individual):
        ## change resource 
        alloc = random.choice(ind.task_allocation)
        choice = [-5,-3,-2,-1,1,2,3,5]
        new_alloc = alloc['resources']['cpu'] + random.choice(choice)
        
        if new_alloc <= 0:
            alloc['resources']['cpu'] = 1
        else:
            alloc['resources']['cpu'] = new_alloc
    
    def mutate_seq(self,ind:individual):
        ## change task sequence
        index1 = random.randrange(len(ind.task_allocation))
        index2 = random.randrange(len(ind.task_allocation))
        while index2 == index1:
            index2 = random.randrange(len(ind.task_allocation))
            
        ind.task_allocation[index1], ind.task_allocation[index2] = ind.task_allocation[index2], ind.task_allocation[index1]
    
    
    def crossover_arith_ave(self, ind1:individual, ind2:individual):
        task_avg = [None]*len(ind1.task_allocation)
        for i in range(len(ind1.task_allocation)):
            name = ind1.task_allocation[i]['name']
            task_id = ind1.task_allocation[i]['task_id']
            task_avg[i] = {
                "name": name,
                "task_id": task_id,
                "resources":{
                    "cpu": (ind1.get_resources(name,task_id)['cpu']+ind2.get_resources(name,task_id)['cpu'])//2
            }}
        ind1.task_allocation = task_avg
    
    def list_dict_found(self, list_dic, dic):
        for i in range(len(list_dic)):
            if list_dic[i]['task_id'] == dic['task_id'] and list_dic[i]['name']== dic['name']:
                return True
        return False

    def list_dict_index(self, list_dic, dic):
        for i in range(len(list_dic)):
            if list_dic[i]['task_id'] == dic['task_id'] and list_dic[i]['name']== dic['name']:
                return i
        return None

    def crossover_pmx(self, ind:individual, ind2:individual):
        size = len(ind.task_allocation)
        p1, p2 = [0]*size, [0]*size
        
        cxpoint1 = random.randint(0, size-1)
        cxpoint2 = random.randint(0, size-1)
        if cxpoint2 < cxpoint1:
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1
        
        # print(cxpoint1,cxpoint2)
        for i in range(cxpoint1,cxpoint2+1):
            p1[i] = ind2.task_allocation[i]
            p2[i] = ind.task_allocation[i]
            
        for i in range(size):
            if i < cxpoint1 or i > cxpoint2:
                ii = ind.task_allocation[i]
                while self.list_dict_found(p1[cxpoint1:cxpoint2+1],ii):
                    # ii = ind.task_allocation[p1[cxpoint1:cxpoint2+1].index(ii)]
                    ii = ind.task_allocation[self.list_dict_index(ind2.task_allocation,ii)]
                p1[i] = ii
                
                ii = ind2.task_allocation[i]
                while self.list_dict_found(p2[cxpoint1:cxpoint2+1],ii):
                    # ii = ind2.task_allocation[p2[cxpoint1:cxpoint2+1].index(ii)]
                    ii = ind2.task_allocation[self.list_dict_index(ind.task_allocation,ii)]
                p2[i] = ii
        
        ind.task_allocation = p1
        ind2.task_allocation = p2
        
    def opt1(self, ind:individual):
        ## add resources for longest task
        task = max(ind.predict_run_seq, key=lambda x:x['total_runtime'])
        index = self.list_dict_index(ind.task_allocation,task)
        new_alloc = random.choice([1,2,3,4,5]) + ind.task_allocation[index]['resources']['cpu']
        if new_alloc <= ind.total_resources['cpu']//2:
            ind.task_allocation[index]['resources']['cpu'] = new_alloc
        
    def opt2(self, ind:individual):
        ## advance the latest task order
        task = max(ind.predict_run_seq, key=lambda x:x['finish_time'])
        index = self.list_dict_index(ind.task_allocation,task)
        if index <= 0:
            return
        new_index = random.randrange(0, index)
        
        element = ind.task_allocation.pop(index)
        ind.task_allocation.insert(new_index, element)
            
    def process_individual(self,ind1,ind2,crossover_rate, mutation_rate):
        if random.random() < 0.5:
            if random.random() < mutation_rate:
                self.mutate_cpu(ind1)
            elif random.random() < mutation_rate:
                self.mutate_seq(ind1)
                
            if random.random() < crossover_rate/2:
                self.crossover_pmx(ind1,ind2)
            elif random.random() < crossover_rate/2:
                self.crossover_arith_ave(ind1,ind2)
            
        else:
            self.opt1(ind1)
            self.opt2(ind1)
        
            
    def run_ga(self, pop_size, num_generations):
        # resources = self.get_resources()
        population = self.generate_population(pop_size)
        # self.his_population.update(population)
        scores = [self.fitness(ind) for ind in population]
        population = [population[i] for i in np.argsort(scores)[::-1]]
        for gen in range(num_generations):
            # population=population[::-1]
            population = population[:pop_size]
            # population = population[pop_size // 2:] + [ind.copy() for ind in population[pop_size // 2:]]
            population.extend([ind.copy() for ind in population[:pop_size // 2]])
            next_population = population[:pop_size // 2]
            futures = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                size = len(next_population)
                for i in range(size//2):
                    futures.append(executor.submit(self.process_individual(next_population[i],next_population[size-i-1],0.8,0.8)))
            concurrent.futures.wait(futures)
            # population = [future.result() for future in futures]
            
            scores = [self.fitness(ind) for ind in population]
            population = [population[i] for i in np.argsort(scores)[::-1]]
            print(f"Generation {gen}: {population[0].score}")
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
    
    # ## test evo1
    # task_submit=[]
    # ga = evosch(task_queue_audit, length_times, core_times,total_cpu=64,at=available_task({"simulation":[i for i in range(24)]}))
    # print(ga.at.get_available_task_id('simulation'))
    # best_individual = ga.run_ga(10, 5)
    # task_submit.extend(best_individual.task_allocation)
    # print(task_submit)
    # ## sum cores
    # total_cpu = 0
    # for task in task_submit:
    #     total_cpu += task['resources']['cpu']
    # print(total_cpu)
    # print(sum(best_individual.tasks_nums.values()))
    # pass
    
    ## test evo2
    ga = evosch2(task_queue_audit, length_times, core_times,resources={'cpu':64,'gpu':4},at=available_task({"simulation":[i for i in range(10)]}))
    # print(ga.at.get_available_task_id('simulation'))
    # print(ga.at.get_all())
    # print(ga.at.get_task_nums())
    # pop = ga.generate_population(2)
    # print(pop[1].total_resources['cpu'])
    
    # completion_time = ga.calculate_completion_time(pop[1])
    # total_time = ga.calculate_total_time(pop[1])
    # print(completion_time)
    # print(total_time)
    # print(pop[1])
    # print(pop[0])
    
    # print(pop[1].task_allocation)
    # ga.mutate_cpu(pop[1])
    # print(pop[1].task_allocation)
    # ga.mutate_seq(pop[1])
    # print(pop[1].task_allocation)
    # print([pop[0].task_allocation[i]['task_id'] for i in range(len(pop[0].task_allocation))])
    # print([pop[1].task_allocation[i]['task_id'] for i in range(len(pop[1].task_allocation))])
    
    # ga.crossover_pmx(pop[0],pop[1])
    # print([pop[0].task_allocation[i]['task_id'] for i in range(len(pop[0].task_allocation))])
    # print([pop[1].task_allocation[i]['task_id'] for i in range(len(pop[1].task_allocation))])
    
    # print(pop[0].task_allocation)
    
    # ga.run_ga(10, 5)
    # pop = ga.generate_population(2)
    # print(ga.fitness(pop[0]))
    
    best_ind = ga.run_ga(100, 200)
    print(best_ind.task_allocation)
    print(ga.fitness(best_ind))
    
    # pop = ga.generate_population(2)
    # print(len(pop[0].task_allocation))
    # print(ga.fitness(pop[0]))
    # print(pop[0].predict_run_seq)
    # ga.opt1(pop[0])
    # print(pop[0].predict_run_seq)
        
        
    # population = ga.generate_population(10)
    # scores = [ga.fitness(ind) for ind in population]
    # population = [population[i] for i in np.argsort(scores)[::-1]]
    # scores = [ga.fitness(ind) for ind in population]
    # print(scores)
    
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
