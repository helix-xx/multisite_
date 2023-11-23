from functools import partial, update_wrapper
from threading import Event, Lock
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from random import shuffle, sample
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import asdict
import hashlib
import logging
import argparse
import shutil
import json
import sys

import ase
from ase.db import connect
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from colmena.models import Result
from colmena.queue import ColmenaQueues
from colmena.queue.redis import RedisQueues
from colmena.thinker import BaseThinker, event_responder, result_processor, ResourceCounter, task_submitter
import proxystore as ps
import numpy as np
import torch
from proxystore.store import register_store
from proxystore.store.file import FileStore
from proxystore.store.globus import GlobusStore, GlobusEndpoints
from proxystore.store.redis import RedisStore
from proxystore.store.utils import get_key

from fff.learning.gc.ase import SchnetCalculator
from fff.learning.gc.functions import GCSchNetForcefield
from fff.learning.gc.models import SchNet, load_pretrained_model
from fff.learning.util.messages import TorchMessage
from fff.sampling.md import MolecularDynamics
from fff.simulation import run_calculator
from fff.simulation.utils import read_from_string, write_to_string

from config import csecluster1 as make_config

logger = logging.getLogger('main')


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


@dataclass
class SimulationTask:
    atoms: ase.Atoms  # Structure to be run
    traj_id: int  # Which trajectory this came from
    ml_eng: float  # Energy predicted from machine learning model
    ml_std: Optional[float] = None  # Uncertainty of the model
    
    
class Thinker(BaseThinker):
    """Class that schedules work on the HPC"""
    
    def __init__(
        self,
        queues: ColmenaQueues,
        out_dir: str,
        train_input: str,
        sampling_input: str,
        simulation_input: str,
        inference_input: str,
    ):
        """
        Args:
        """
        self.start_training = Event()
        self.start_sampling = Event()
        self.start_simulation = Event()
        self.start_inference = Event()
        ## data prepare here
        ## trigger event here
        pass
    
    @event_responder(event_name='start_training')
    def train_models(self):
        """Submit the models to be retrained"""
    @result_processor(topic_name='train')
    
        
    @event_responder(event_name='start_sampling')
    def sample_structures(self):
        """Sample new structures"""
    @result_processor(topic_name='sample')
        
    @event_responder(event_name='start_simulation')
    def run_simulations(self):
        """Run the simulations"""
    @result_processor(topic_name='simulation')
        
    @event_responder(event_name='start_inference')
    def run_inference(self):
        """Run the inference"""
    @result_processor(topic_name='inference')