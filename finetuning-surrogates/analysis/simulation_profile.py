from fff.simulation import run_calculator
from fff.simulation.utils import read_from_string, write_to_string
from _pytest.fixtures import fixture
from ase.build import molecule
# from memory_profiler import profile
calc = dict(calc='psi4', method='pbe0-d3', basis='aug-cc-pvdz', num_threads=8)
import os
import psutil
from functools import partial, update_wrapper


temp_path = "./temp"
## print complete whole path
path = os.path.abspath(temp_path)
print(path)

# @profile
def _wrap(func, **kwargs):
    out = partial(func, **kwargs)
    update_wrapper(out, func)
    return out

# @fixture()
def atoms():
    return molecule('H2O')
atom = atoms()

# @fixture()
def cluster():
    xyz = """30

O       7.581982610000000     -0.663324770000000      5.483883860000000
H       8.362350460000000     -0.079370470000000      5.498567580000000
H       7.846055030000000     -1.464757200000000      5.041030880000000
O       9.456702229999999      1.642301080000000      8.570644379999999
H      10.114471399999999      1.655581000000000      9.261547090000001
H       9.181962009999999      2.562770840000000      8.428308489999999
O       9.664885520000000      1.027763610000000      5.758778100000000
H       9.485557560000000      1.914335850000000      5.411871910000000
H       9.760457990000001      1.144007330000000      6.710969450000000
O       6.000383380000000      4.009448050000000      7.349214080000000
H       5.983903880000000      4.025474550000000      6.383275510000000
H       5.536083220000000      3.203337670000000      7.608772750000000
O       4.833731170000000      1.482195020000000      7.883007530000000
H       5.628127100000000      0.955450120000000      8.084721569999999
H       4.134047510000000      1.149705890000000      8.438218120000000
O       7.110025880000000      0.051394890000000      8.205573080000001
H       7.262372970000000     -0.325556960000000      7.328944680000000
H       7.906465050000000      0.552607360000000      8.419908520000000
O       6.173881530000000      3.688445090000000      4.528872010000000
H       5.701079370000000      4.022632120000000      3.771772860000000
H       5.903837200000000      2.759263990000000      4.641063690000000
O       5.429551600000000      1.145089270000000      5.097751140000000
H       6.135000710000000      0.486118580000000      5.133624550000000
H       5.085167410000000      1.211727260000000      5.997292520000000
O       8.597597120000000      4.222480770000000      8.031750680000000
H       7.641802790000000      4.166800020000000      7.848542690000000
H       8.760176660000001      5.097825050000000      8.372748370000000
O       8.954336169999999      3.647526740000000      5.177083970000000
H       8.927373890000000      3.954237700000000      6.090421680000000
H       8.043264389999999      3.698852060000000      4.860042570000000
"""
    return read_from_string(xyz, 'xyz')
# xyz = cluster()
# xyz = write_to_string(xyz, 'xyz')
print(atom)
print(len(atom))
# print(atom.get_potential_energies())
# xyz = write_to_string(atom, 'xyz')
# print(xyz)

# my_run_simulation = _wrap(run_calculator, calc=calc, temp_path=path)
# value = my_run_simulation(xyz)
# print(value)
# atom = read_from_string(value, 'json')
# xyz = write_to_string(atom, 'xyz')
# print(xyz)
# print(atom)
# print(atom.get_potential_energies())