"""Collections of Python functions for generating new training data"""
from concurrent.futures import ProcessPoolExecutor
import re
from tempfile import TemporaryDirectory
import tempfile
import shutil
from typing import Optional, Union
import os

from ase.calculators.calculator import Calculator
from ase.calculators.psi4 import Psi4

from fff.simulation.utils import write_to_string, read_from_string
# from memory_profiler import profile

CalcType = Union[Calculator, dict]

# @profile
# def run_calculator(xyz: str, calc: CalcType, temp_path: Optional[str] = None) -> str:
#     """Run an NWChem computation on the requested cluster

#     Args:
#         xyz: Cluster to evaluate
#         calc: ASE calculator to use. Either the calculator object or a dictionary describing the settings
#             (only Psi4 supported at the moment for dict)
#         temp_path: Base path for the scratch files
#     Returns:
#         Atoms after the calculation in a JSON format
#     """

#     # Some calculators do not clean up their resources well
#     with ProcessPoolExecutor(max_workers=1) as exe:
#         fut = exe.submit(_run_calculator, str(xyz), calc, temp_path)  # str ensures proxies are resolved
#         return fut.result()
    
def run_calculator(xyz: str, calc: CalcType, temp_path: Optional[str] = None, cpu:int = 1,gpu:int = 0) -> str:
    """Run an NWChem computation on the requested cluster

    Args:
        xyz: Cluster to evaluate
        calc: ASE calculator to use. Either the calculator object or a dictionary describing the settings
            (only Psi4 supported at the moment for dict)
        temp_path: Base path for the scratch files
    Returns:
        Atoms after the calculation in a JSON format
    """
    if calc["num_threads"] != cpu:
        calc["num_threads"] = cpu
    # Some calculators do not clean up their resources well
    with ProcessPoolExecutor(max_workers=1) as exe:
        fut = exe.submit(_run_calculator, str(xyz), calc, temp_path)  # str ensures proxies are resolved
        return fut.result()


def _run_calculator(xyz: str, calc: CalcType, temp_path: Optional[str] = None) -> str:
    """Runs the above function, designed to be run inside a new Process"""

    # Parse the atoms object
    atoms = read_from_string(xyz, 'xyz')

    with TemporaryDirectory(dir=temp_path, prefix='fff') as temp_dir:
        # Execute from the temp so that the calculators do not interfere

        # Special case for Psi4 which sets the run directory on creating the object
        if isinstance(calc, dict):
            calc = calc.copy()
            assert calc.pop('calc') == 'psi4', 'only psi4 is supported for now'
            # calc = Psi4(**calc, directory=temp_dir, PSI_SCRATCH=temp_path)
            # print(f"temp_dir {temp_dir}, temp_path {temp_path}")
            calc = Psi4(**calc, directory=temp_dir, PSI_SCRATCH=temp_path)

        # Run the calculation
        atoms.calc = calc
        try:
            atoms.get_forces()
            atoms.get_potential_energy()
        except BaseException as exc:
            raise ValueError(f'Calculation failed: {exc}')

        # manually clean
        pid = os.getpid()
        pattern = r"psi.*" + str(pid) + r".*"
        for filename in os.listdir(temp_path):
            # 使用正则表达式匹配文件名
            if re.match(pattern, filename):
                # 构建文件路径
                file_path = os.path.join(temp_path, filename)
                # 删除文件
                os.remove(file_path)
        
        # Convert it to JSON
        return write_to_string(atoms, 'json')
    
    
    ### modified by yxx
    ## due to nfs file system, DemporaryDirectory rmtree fail by "directory not empty"
    ## mkdir and rmtree manually
    # with TemporaryDirectory(dir=temp_path, prefix='fff') as temp_dir:
    # temp_dir = tempfile.mkdtemp(dir=temp_path,prefix='fff')
    # # Execute from the temp so that the calculators do not interfere
    # os.chdir(temp_dir)

    # # Special case for Psi4 which sets the run directory on creating the object
    # if isinstance(calc, dict):
    #     calc = calc.copy()
    #     assert calc.pop('calc') == 'psi4', 'only psi4 is supported for now'
    #     calc = Psi4(**calc, directory=temp_dir, PSI_SCRATCH=temp_path)

    # # Run the calculation
    # atoms.calc = calc
    # try:
    #     atoms.get_forces()
    #     atoms.get_potential_energy()
    # except BaseException as exc:
    #     raise ValueError(f'Calculation failed: {exc}')

    # shutil.rmtree(temp_dir,ignore_errors=True)
    # Convert it to JSON
    # return write_to_string(atoms, 'json')
    