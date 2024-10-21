import sys, os, glob
import numpy as np
from numba import njit
from qcp_numba import CalcRMSDRotationalMatrix
import pandas as pd
import multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

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
        x = rotmat[0] * frag[0][i] + rotmat[1] * frag[1][i] + rotmat[2] * frag[2][i]
        y = rotmat[3] * frag[0][i] + rotmat[4] * frag[1][i] + rotmat[5] * frag[2][i]
        z = rotmat[6] * frag[0][i] + rotmat[7] * frag[1][i] + rotmat[8] * frag[2][i]
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


def getCAcoo(frag):
    """
    Parse CA coordinates from the numpy coordinate array
        :param frag:
        :return:
    """

    x = frag[0][2::6]
    y = frag[1][2::6]
    z = frag[2][2::6]
    return np.array([x, y, z])


def getXcoo(frag, atom_type):
    """
    Parse coordinates given the type of backbone atom
    # atom_type = ['N', 'H', 'CA', 'C', 'O'] old
    # atom_type = ['N', 'H', 'CA', 'C', 'O', 'CB] new
    Args:
        frag:

    Returns:

    """
    #repeat_factor = 5 #old database had only 5 backbone elements
    repeat_factor = 6
    x = frag[0][atom_type::repeat_factor]
    y = frag[1][atom_type::repeat_factor]
    z = frag[2][atom_type::repeat_factor]
    return np.array([x, y, z])

class NumbaQCP:
    """
    This is Numba-Numpy adaptation of Original C-code taken from "Rapid calculation of RMSD using a quaternion-based characteristic
    polynomial". The Numba-Numpy implementation is 4x faster than Python-SWIG-C  wrapping.

 *    If you use this QCP rotation calculation method in a publication, please
 *    reference:
 *
 *      Douglas L. Theobald (2005)
 *      "Rapid calculation of RMSD using a quaternion-based characteristic
 *      polynomial."
 *      Acta Crystallographica A 61(4):478-480.
 *
 *      Pu Liu, Dmitris K. Agrafiotis, and Douglas L. Theobald (2009)
 *      "Fast determination of the optimal rotational matrix for macromolecular
 *      superpositions."
 *      Journal of Computational Chemistry 31(7):1561-1563.
 *
 *  Copyright (c) 2009-2016 Pu Liu and Douglas L. Theobald
 *  All rights reserved.
 *  Redistribution and use in source and binary forms, with or without modification, are permitted
 *  provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice, this list of
 *    conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice, this list
 *    of conditions and the following disclaimer in the documentation and/or other materials
 *    provided with the distribution.
 *  * Neither the name of the <ORGANIZATION> nor the names of its contributors may be used to
 *    endorse or promote products derived from this software without specific prior written
 *    permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 *  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """

    def __init__(self, previous_smotif, csmotif, direction, cutoff, previous_sse_index):
        self.previous_smotif = previous_smotif
        self.csmotif = csmotif
        self.direction = direction
        self.cutoff = cutoff
        self.previous_sse_index = previous_sse_index

    def compute_rmsd(self):

        psmotif = (self.previous_smotif[2][1])[:]
        psmotif_index = self.previous_sse_index[-1]

        if self.direction == 'left':
            frag_a = np.empty_like((psmotif[psmotif_index])[0:3])
            frag_a[:] = (psmotif[psmotif_index])[0:3]
            frag_b = (self.csmotif[2])[0:3]
            native_fragb_2ndsse = (self.csmotif[1])[0:3]
        else:

            frag_a = np.empty_like((psmotif[psmotif_index])[0:3])
            frag_a[:] = (psmotif[psmotif_index])[0:3]
            frag_b = (self.csmotif[1])[0:3]
            native_fragb_2ndsse = (self.csmotif[2])[0:3]

        frag_a, a_cen = centerCoo(frag_a)
        frag_b, b_cen = centerCoo(frag_b)

        frag_aca = getCAcoo(frag_a)
        frag_bca = getCAcoo(frag_b)

        fraglen = len(frag_aca[0])

        rotation_matrix, rmsd = CalcRMSDRotationalMatrix(frag_aca, frag_bca, fraglen)

        if rmsd > self.cutoff:
            return rmsd, []

        # translate the other SSE of the current smotif

        cm_sse2nd = translateCM(native_fragb_2ndsse, b_cen, len(native_fragb_2ndsse[0]))
        rot_sse_2nd = applyRot(cm_sse2nd, rotation_matrix, len(cm_sse2nd[0]))
        trans_sse2nd = applyTranslation(rot_sse_2nd, a_cen, len(rot_sse_2nd[0]))

        # append the translated coordinates

        if self.direction == 'left':
            temp_holder = self.csmotif[1]
            temp_holder[0:3] = trans_sse2nd
        else:
            temp_holder = self.csmotif[2]
            temp_holder[0:3] = trans_sse2nd

        if self.direction == 'left':
            psmotif.insert(0, temp_holder)
        else:
            psmotif.append(temp_holder)

        return rmsd, psmotif

class NumbaQcpRMSD:
    
    def __init__(self, reference_pdb, target_pdb):
        self.reference_pdb = reference_pdb
        self.target_pdb = target_pdb
        
        
    def parse_coors(self, pdb_file):
        
        with open(pdb_file, 'r') as pdb:
            lines = pdb.readlines()
            
        # parse the coordinates of the Cα atoms from a given chain and range
        vector = []
        for line in lines:
            if line.startswith('ATOM') and line[13:15] == 'CA':
                vector.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])

        # calculate the centroid
        coor_array = np.array(vector)        
        return coor_array
                
    def compute_rmsd(self):
        reference_ca = self.parse_coors(self.reference_pdb)
        target_ca = self.parse_coors(self.target_pdb)
        
        rotation_matrix, rmsd = CalcRMSDRotationalMatrix(reference_ca, target_ca, len(reference_ca))        
        return rmsd
    
    
if __name__ == '__main__':
    pdbfile_path = sys.argv[1]    
    pdb_files = glob.glob(pdbfile_path + '*.pdb')
    #pdb_files = pdb_files[:100]
    # create a pandas dataframe to store the rmsd values
    rmsd_df = pd.DataFrame(columns=['pdb1', 'pdb2', 'rmsd'])    
    # compute all pairwise rmsd values for the pdb files and store them in the dataframe.
    # compute the rmsd values in parallel
    # use progress bar to track the progress
    
    pool = mp.Pool(mp.cpu_count())
    rmsd_values = pool.starmap(NumbaQcpRMSD, [(pdb_files[i], pdb_files[j]) for i in range(len(pdb_files)) for j in range(i+1, len(pdb_files))])
    pool.close()
    rmsd_df['pdb1'] = [os.path.basename(x.reference_pdb).split('.')[0] for x in rmsd_values]
    rmsd_df['pdb2'] = [os.path.basename(x.target_pdb).split('.')[0] for x in rmsd_values]
    rmsd_df['rmsd'] = [x.compute_rmsd() for x in rmsd_values]
    rmsd_df.to_csv('rmsd_values.csv', index=False)
    
    # plot disance map
    distance_map = np.zeros((len(pdb_files), len(pdb_files)))
    for i in range(len(pdb_files)):
        for j in range(i+1, len(pdb_files)):
            distance_map[i, j] = rmsd_df[(rmsd_df['pdb1'] == os.path.basename(pdb_files[i]).split('.')[0]) & (rmsd_df['pdb2'] == os.path.basename(pdb_files[j]).split('.')[0])]['rmsd'].values[0]
    
    distance_map = distance_map + distance_map.T
    distance_map[distance_map == 0] = np.nan
    plt.imshow(distance_map)
    plt.colorbar()
    plt.savefig('distance_map.png')
    plt.close()
    
        
    
            
    