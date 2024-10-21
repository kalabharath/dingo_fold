import sys
sys.path.append('./numba_rmsd/')
import numpy as np
from numba import njit
from qcp_numba import CalcRMSDRotationalMatrix
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
# disable warnings
import warnings
warnings.filterwarnings("ignore")


@njit
def centerCoo(coo_array):
    """
    simplified numpy implementation would be
    R -= R.sum(0) / len(R)
    if True:
        return R, R
    however, the Numba execution is faster
    :param coo_array:
    :return:
    """

    xsum, ysum, zsum = 0, 0, 0

    for i in range(0, len(coo_array[0])):
        xsum += coo_array[0][i]
        ysum += coo_array[1][i]
        zsum += coo_array[2][i]

    xsum /= len(coo_array[0])
    ysum /= len(coo_array[0])
    zsum /= len(coo_array[0])

    for i in range(0, len(coo_array[0])):
        coo_array[0][i] -= xsum
        coo_array[1][i] -= ysum
        coo_array[2][i] -= zsum

    return coo_array, [xsum, ysum, zsum]

@njit
def applyRot(frag, rotmat, fraglen):
    """

    Args:
        frag:
        rotmat:
        fraglen:

    Returns:

    """
    for i in range(0, fraglen):
        x = rotmat[0] * frag[0][i] + rotmat[1] * \
            frag[1][i] + rotmat[2] * frag[2][i]
        y = rotmat[3] * frag[0][i] + rotmat[4] * \
            frag[1][i] + rotmat[5] * frag[2][i]
        z = rotmat[6] * frag[0][i] + rotmat[7] * \
            frag[1][i] + rotmat[8] * frag[2][i]
        frag[0][i] = x
        frag[1][i] = y
        frag[2][i] = z
    return frag

@njit
def translateCM(coo_array, cm, fraglen):
    """

    Args:
        coo_array:
        cm:
        fraglen:

    Returns:

    """

    for i in range(0, fraglen):
        coo_array[0][i] = coo_array[0][i] - cm[0]
        coo_array[1][i] = coo_array[1][i] - cm[1]
        coo_array[2][i] = coo_array[2][i] - cm[2]
    return coo_array

@njit
def applyTranslation(frag, cen_mass, fraglen):
    """

    Args:
        frag:
        cen_mass:
        fraglen:

    Returns:

    """

    for i in range(0, fraglen):
        frag[0][i] += cen_mass[0]
        frag[1][i] += cen_mass[1]
        frag[2][i] += cen_mass[2]
    return frag


def parse_coors_dict(pdb_file, resi_range):
    
    with open(pdb_file, 'r') as pdb:
        lines = pdb.readlines()
    # parse the coordinates of the Cα atoms from a given chain and range
    ca_coor_array = {}

    for line in lines:
        if line.startswith('ATOM') and line[12:16].strip() == 'CA':
            resi_num = int(line[22:26].strip())
            if resi_num in resi_range:
                ca_coor_array[resi_num] = [float(line[30:38].strip()), float(
                    line[38:46].strip()), float(line[46:54].strip())]
    return ca_coor_array

def dict2array(coor_dict):
    ref_array = []
    # get keys from coor_dict
    resi_numbers = list(coor_dict.keys())
    # sort them
    resi_numbers.sort()
    # iterate through resi_numbers and get the coors from coor_dict and append them to ref_array
    for resi in resi_numbers:
        ref_array.append(coor_dict[resi])
      
    ref_array = np.array(ref_array)    
    # the shape of ref_array is (n, 3), where n is the number of residues, convert it to (3, n)
    ref_array = np.transpose(ref_array)    
    return ref_array


def calc_smotif_rmsd(pdb1, pdb2, coor_range):      
    ref_sse1_dict = parse_coors_dict(pdb_file=pdb1, resi_range=list(range(coor_range[0], coor_range[1]+1)))
    ref_sse2_dict = parse_coors_dict(pdb_file=pdb1, resi_range=list(range(coor_range[2], coor_range[3]+1)))    
    ref_dict = {**ref_sse1_dict, **ref_sse2_dict}
    
    target_sse1_dict = parse_coors_dict(pdb_file=pdb2, resi_range=list(range(coor_range[4], coor_range[5]+1)))
    target_sse2_dict = parse_coors_dict(pdb_file=pdb2, resi_range=list(range(coor_range[6], coor_range[7]+1)))
    target_dict = {**target_sse1_dict, **target_sse2_dict}
    
    ref_array = dict2array(ref_dict)
    target_array = dict2array(target_dict)
    
    ref_array, center_ref = centerCoo(ref_array)
    target_array, target_ref = centerCoo(target_array)
    if len(ref_array[0]) != len(target_array[0]):
        #print (f"The number of residues in the two structures are not equal")
        return None        
    else:
        rot1, rmsd = CalcRMSDRotationalMatrix(
            ref_array, target_array, len(ref_array[0]))
        return rmsd


