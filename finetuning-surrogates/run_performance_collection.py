from functools import partial, update_wrapper
from threading import Event, Lock
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from random import shuffle, sample
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import asdict
import hashlib
import logging
import argparse
import shutil
import json
import sys
import os
import pickle
import time
import uuid

import ase
from ase.db import connect
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.calculators.calculator import Calculator
from colmena.models import Result, ResourceRequirements
from colmena.queue import ColmenaQueues
from colmena.queue.redis import RedisQueues
from colmena.thinker import (
    BaseThinker,
    event_responder,
    result_processor,
    ResourceCounter,
    task_submitter,
)
import proxystore as ps
import numpy as np
import torch
import torch.multiprocessing as mp

from fff.learning.gc.ase import SchnetCalculator
from fff.learning.gc.functions import GCSchNetForcefield
from fff.learning.gc.models import SchNet, load_pretrained_model
from fff.learning.util.messages import TorchMessage
from fff.sampling.md import MolecularDynamics
from fff.simulation import run_calculator
from fff.simulation.utils import read_from_string, write_to_string


    
@dataclass
class SimulationTask:
    atoms: ase.Atoms  # Structure to be run
    traj_id: int  # Which trajectory this came from
    ml_eng: float  # Energy predicted from machine learning model
    ml_std: Optional[float] = None  # Uncertainty of the model

class PerformanceDataCollector(BaseThinker):
    """收集任务性能数据的简化版Thinker"""
    
    def __init__(
        self,
        queues: ColmenaQueues,
        out_dir: Path,
        task_queue_audit: list,  # 预定义的simulation任务列表
        cpu_configs: list = [1, 2, 4, 8, 12, 16, 20],  # 要测试的CPU配置
        gpu_configs: list = [1, 2, 3, 4],  # 要测试的GPU配置
        samples_per_config: int = 2,  # 每个配置重复次数
        starting_model: SchNet = None,  # 起始模型
        my_logger: logging.Logger = None,
    ):
        super().__init__(queues)
        self.out_dir = out_dir
        self.task_queue_audit = task_queue_audit
        self.cpu_configs = cpu_configs
        self.gpu_configs = gpu_configs
        self.samples_per_config = samples_per_config
        self.expansion_times = 3
        
        # 性能数据存储
        self.performance_data = defaultdict(list)
        
        simulation_times = 50
        train_times = 2
        if len(task_queue_audit) >=simulation_times:
            self.logger.info(f"Truncating task queue to {simulation_times} tasks")
            self.task_queue_audit = task_queue_audit[:simulation_times]
        
        self.total_simulation_tasks = simulation_times * len(cpu_configs) * samples_per_config
        self.total_train_tasks = len(gpu_configs) * samples_per_config * self.expansion_times
        # 任务计数
        self.completed_tasks = 0
        # self.total_tasks = self.total_simulation_tasks + self.total_train_tasks
        self.total_tasks = self.total_simulation_tasks
        self.simulation_sunmitted = 0
        self.train_submitted = 0
        self.logger.info(f'Total tasks: {self.total_tasks}, simulation tasks: {self.total_simulation_tasks}, training tasks: {self.total_train_tasks}')
        
        self.starting_model_proxy = starting_model
        self.db_path = os.path.expanduser('~/project/colmena/multisite_/data/forcefields/starting-model/initial-database_test.db')        # Load in the training dataset
        with connect(self.db_path) as db:
            self.logger.info(
                f'Connected to a database with {len(db)} entries at {self.db_path}'
            )
            self.all_examples = np.array([x.toatoms() for x in db.select("")], dtype=object)
        self.logger.info(f'Loaded {len(self.all_examples)} training examples')
        
        # self.logger = logger
        
        # 创建输出目录
        perf_dir = out_dir / 'performance_data'
        perf_dir.mkdir(exist_ok=True)
        
    @task_submitter(task_type="simulate", enable_allocate=False)
    def submit_simulation_tasks(self, **kwargs):
        """提交simulation任务"""
        self.logger.info(f'Submitting simulation tasks')
        if self.simulation_sunmitted >= self.total_simulation_tasks:
            time.sleep(60)
            return
        for task in self.task_queue_audit:
            for cpu in self.cpu_configs:
                for _ in range(self.samples_per_config):
                    atoms = task.atoms
                    atoms.set_center_of_mass([0, 0, 0])
                    xyz = write_to_string(atoms, 'xyz')
                    
                    self.queues.send_inputs(
                        xyz,
                        method='run_calculator',
                        topic='simulate',
                        keep_inputs=True,
                        task_info={
                            'cpu_config': cpu,
                            'gpu_config': 0,
                            'xyz': xyz,
                        },
                        resources=ResourceRequirements(
                            cpu=cpu,
                            gpu=0,
                            node='all'
                        ),
                    )
                    self.simulation_sunmitted += 1
    
    # @task_submitter(task_type="train", enable_allocate=False)
    # def submit_training_tasks(self, **kwargs):
    #     """提交training任务"""
    #     self.logger.info(f'Submitting training tasks')
    #     if self.train_submitted >= self.total_train_tasks:
    #         time.sleep(60)
    #         return
    #     expansion_size = 25
    #     for expansion in range(self.expansion_times):
    #         all_examples = self.all_examples
    #         # all_examples.extend(all_examples[expansion * 25:(expansion + 1) * 25])
    #         new_examples = all_examples[expansion * expansion_size:(expansion + 1) * expansion_size]
    #         all_examples = np.concatenate([all_examples, new_examples])
    #         n_train = int(len(all_examples) * 0.9)
    #         shuffle(all_examples)
    #         train_sets = [all_examples[:n_train]]
    #         valid_sets = [all_examples[n_train:]]
    #         for gpu in self.gpu_configs:
    #             for _ in range(self.samples_per_config):
    #                 self.queues.send_inputs(
    #                     self.starting_model_proxy,
    #                     train_sets[0],
    #                     valid_sets[0],
    #                     method='train',
    #                     topic='train',
    #                     keep_inputs=True,
    #                     task_info={
    #                         'cpu_config': 1,
    #                         'gpu_config': gpu,
    #                     },
    #                     resources=ResourceRequirements(
    #                         cpu=1,
    #                         gpu=gpu,
    #                         node='all'
    #                     ),
    #                 )
    #                 self.train_submitted += 1
    
    @result_processor(topic='simulate')
    def store_simulation_result(self, result: Result):
        """处理simulation任务的结果"""
        self.logger.info(f'Received result for simulation task {result.task_id}, success={result.success}')
        if result.success:
            self.performance_data['simulation'].append({
                'task_id': result.task_id,
                'cpu_config': result.task_info['cpu_config'],
                'gpu_config': result.task_info['gpu_config'],
                'runtime': result.time_running,
            })
            with open(self.out_dir / 'simulation-results.json', 'a') as fp:
                print(result.json(exclude={'value', 'inputs'}), file=fp)

        assert result.success, result.failure_info.exception
        self._check_completion()
    
    @result_processor(topic='train')
    def store_training_result(self, result: Result):
        """处理training任务的结果"""
        self.logger.info(f'Received result for training task {result.task_id}, success={result.success}')
        if result.success:
            self.performance_data['train'].append({
                'task_id': result.task_id,
                'cpu_config': result.task_info['cpu_config'],
                'gpu_config': result.task_info['gpu_config'],
                'runtime': result.time_running,
            })
            with open(self.out_dir / 'training-results.json', 'a') as fp:
                print(result.json(exclude={'inputs', 'value'}), file=fp)
                
        assert result.success, result.failure_info.exception
        self._check_completion()
    
    def _check_completion(self):
        """检查是否所有任务都已完成"""
        self.completed_tasks += 1
        if self.completed_tasks >= self.total_tasks:
            # 保存性能数据
            perf_file = self.out_dir / 'performance_data' / f'perf_data_{time.strftime("%Y%m%d_%H%M%S")}.pkl'
            with open(perf_file, 'wb') as f:
                pickle.dump(self.performance_data, f)
            
            self.done.set()

def setup_logger(out_dir: Path):
    """设置日志系统"""
    handlers = [
        logging.FileHandler(out_dir / 'runtime.log'),
        logging.StreamHandler(sys.stdout),
    ]

    class ParslFilter(logging.Filter):
        """Filter out Parsl debug logs"""
        def filter(self, record):
            return not (record.levelno == logging.DEBUG and '/parsl/' in record.pathname)

    for h in handlers:
        h.addFilter(ParslFilter())

    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=handlers,
    )
    
    
def _wrap(func, **kwargs):
    """包装函数，固定某些参数"""
    out = partial(func, **kwargs)
    update_wrapper(out, func)
    return out

def run_performance_collection(args):
    """运行性能数据收集"""
    # 准备输出目录
    start_time = datetime.now()
    params_hash = hashlib.sha256(json.dumps(args.__dict__).encode()).hexdigest()[:6]
    out_dir = Path(args.work_dir) / f'perf_collection_{start_time.strftime("%Y%m%d_%H%M%S")}-{params_hash}'
    out_dir.mkdir(parents=True)
    
    # 设置日志
    setup_logger(out_dir)
    logger = logging.getLogger('main')
    logger.info(f'Run directory: {out_dir}')
    
    # 创建执行器配置
    from my_util.multi_node_config import create_executor_from_config
    config, node_resources = create_executor_from_config(
        args.work_dir + "/resources.ini", 
        str(out_dir)
    )
    
    # 保存运行参数
    with open(out_dir / 'runparams.json', 'w') as fp:
        json.dump(args.__dict__, fp)
        
    # 设置计算方法
    calc = dict(calc='psi4', method='pbe0-d3', basis='aug-cc-pvdz', num_threads=8)
    my_run_simulation = _wrap(
        run_calculator,
        calc=calc,
        temp_path=os.path.expanduser('~/project/colmena/multisite_/finetuning-surrogates/psi4'),
    )
    starting_model = torch.load(args.starting_model, map_location='cpu')
    
    schnet = GCSchNetForcefield()
    my_train_schnet = _wrap(
        schnet.train,
        num_epochs=256,
        patience=8,
        reset_weights=False,
        huber_deltas=[1, 10],
        parallel=2,
    )
    # 定义methods列表
    methods = [my_run_simulation, my_train_schnet]
    
    # 加载预定义的simulation任务
    with open(args.task_queue_path, 'rb') as f:
        task_queue_audit = pickle.load(f)
    logger.info(f'Loaded {len(task_queue_audit)} predefined tasks')
    
    # 设置队列
    from colmena.queue.redis import RedisQueues
    queues = RedisQueues(
        hostname=args.redishost,
        port=args.redisport,
        prefix=start_time.strftime("%d%b%y-%H%M%S"),
        topics=['simulate', 'sample', 'train', 'infer'],
        methods=['run_calculator', 'run_sampling', 'train', 'evaluate'],
        serialization_method='pickle',
        keep_inputs=False,
        scheduler="fcfs",
        available_resources=node_resources,
    )
    
    # 创建任务服务器
    from colmena.task_server import ParslTaskServer
    doer = ParslTaskServer(methods, queues, config)
    
    # 创建性能数据收集器
    collector = PerformanceDataCollector(
        queues=queues,
        out_dir=out_dir,
        task_queue_audit=task_queue_audit,
        cpu_configs=args.cpu_configs,
        gpu_configs=args.gpu_configs,
        samples_per_config=args.samples_per_config,
        starting_model=starting_model,
        my_logger=logger,
    )
    
    try:
        # 启动服务器和收集器
        doer.start()
        collector.start()
        logger.info('Started servers')
        
        # 等待完成
        collector.join()
        logger.info('Collection completed')
    finally:
        queues.send_kill_signal()
        doer.join()
        logger.info('Servers shut down')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task-queue-path', required=True, help='Path to predefined simulation tasks')
    parser.add_argument('--work-dir', required=True, help='Working directory')
    parser.add_argument('--redishost', default='127.0.0.1')
    parser.add_argument('--redisport', default='7485')
    parser.add_argument('--cpu-configs', nargs='+', type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument('--gpu-configs', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--samples-per-config', type=int, default=3)
    parser.add_argument('--starting-model', required=True, help='Path to the starting model')
    
    args = parser.parse_args()
    run_performance_collection(args)