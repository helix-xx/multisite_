from dataclasses import dataclass
from typing import Optional
import ase

@dataclass
class SimulationTask:
    atoms: ase.Atoms  # Structure to be run
    traj_id: int  # Which trajectory this came from
    ml_eng: float  # Energy predicted from machine learning model
    ml_std: Optional[float] = None  # Uncertainty of the model

@dataclass
class my_SimulationTask:
    simu_task: SimulationTask  # basic information store in SimulationTask
    dft_energy: Optional[float] = None  # DFT energy of the structure
    # Dictionary to store DFT run times for different CPU cores
    dft_time: Optional[dict[int, float]] = None
    temp_cores: Optional[int] = None  # Number of CPU cores used for the DFT calculation
    start_time: Optional[int] = None
    
class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class SingletonClass(metaclass=SingletonMeta):
    def __init__(self):
        pass