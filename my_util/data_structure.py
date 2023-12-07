from dataclasses import dataclass
from typing import Optional
import ase

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