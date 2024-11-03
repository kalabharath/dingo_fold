# create a python init file 
# add the following lines to rmsd_clustering/numba_rmsd/__init__.py:
# import from numba_rmsd folder
import sys
sys.path.append('./numba_rmsd/')
from nqcp import *
from qcp_numba import *
