import numpy as np
import os
import glob
from numba_rmsd.nqcp import centerCoo, getCAcoo
from numba_rmsd.qcp_numba import CalcRMSDRotationalMatrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

def get_ca_coords_and_sequence(pdb_file):
    """Extract CA coordinates and sequence from PDB file"""
    aa_dict = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    ca_coords = []
    sequence = []
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                resname = line[17:20].strip()
                
                if atom_name == 'CA':
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    ca_coords.append([x, y, z])
                    
                    if resname in aa_dict:
                        sequence.append(aa_dict[resname])
    
    return np.array(ca_coords).T, ''.join(sequence)

def files_exist(smotif_output_dir):
    """
    Check if all output files exist for a given Smotif directory
    """
    required_files = [
        "rmsd_heatmap.png",
        "rmsd_matrix.csv",
        "redundant_groups.txt"
    ]
    
    return all(os.path.exists(os.path.join(smotif_output_dir, f)) for f in required_files)

def calculate_all_vs_all_rmsd(args):
    """
    Calculate all-vs-all RMSD for PDB files in a directory
    """
    smotif_dir, output_dir, rmsd_cutoff = args
    
    smotif_type = os.path.basename(smotif_dir)
    smotif_output_dir = os.path.join(output_dir, smotif_type)
    
    # Check if this Smotif has already been processed
    if files_exist(smotif_output_dir):
        print(f"Skipping {smotif_type} - already processed")
        return
    
    # Create output directory
    os.makedirs(smotif_output_dir, exist_ok=True)
    
    # Get all PDB files
    pdb_files = glob.glob(os.path.join(smotif_dir, "*.pdb"))
    print(f"Processing {len(pdb_files)} structures in {smotif_type}")
    
    # Store CA coordinates and sequences
    structures = {}
    for pdb_file in pdb_files:
        try:
            coords, sequence = get_ca_coords_and_sequence(pdb_file)
            coords, _ = centerCoo(coords)
            structures[os.path.basename(pdb_file)] = {
                'coords': coords,
                'sequence': sequence
            }
            
        except Exception as e:
            print(f"Error processing {pdb_file}: {str(e)}")
            continue
    
    # Initialize RMSD matrix
    pdb_list = sorted(structures.keys())
    n_structs = len(pdb_list)
    rmsd_matrix = np.zeros((n_structs, n_structs))
    
    # Calculate all-vs-all RMSD
    redundant_groups = []
    processed = set()
    
    for i, pdb1 in enumerate(pdb_list):
        coords1 = structures[pdb1]['coords']
        seq1 = structures[pdb1]['sequence']
        
        for j, pdb2 in enumerate(pdb_list[i:], i):
            coords2 = structures[pdb2]['coords']
            seq2 = structures[pdb2]['sequence']
            
            if coords1.shape != coords2.shape:
                rmsd_matrix[i,j] = rmsd_matrix[j,i] = np.nan
                continue
                
            try:
                rotation_matrix, rmsd = CalcRMSDRotationalMatrix(coords1, coords2, len(coords1[0]))
                rmsd_matrix[i,j] = rmsd_matrix[j,i] = rmsd
                
                # Check both RMSD and sequence identity
                if i != j and rmsd <= rmsd_cutoff and pdb1 not in processed:
                    # Check sequence identity
                    if seq1 == seq2:  # 100% identity
                        if not any(pdb1 in group for group in redundant_groups):
                            redundant_groups.append([pdb1, pdb2])
                        else:
                            for group in redundant_groups:
                                if pdb1 in group and pdb2 not in group:
                                    group.append(pdb2)
                    
            except Exception as e:
                print(f"Error calculating RMSD between {pdb1} and {pdb2}: {str(e)}")
                rmsd_matrix[i,j] = rmsd_matrix[j,i] = np.nan
                continue
        
        processed.add(pdb1)
        if i % 10 == 0:
            print(f"{smotif_type}: Processed {i+1}/{n_structs} structures")
    
    # Save results
    try:
        # Create and save heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(rmsd_matrix, 
                    xticklabels=pdb_list, 
                    yticklabels=pdb_list,
                    cmap='viridis',
                    square=True)
        plt.title(f'RMSD Matrix - {smotif_type}')
        plt.xlabel('Structure')
        plt.ylabel('Structure')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        heatmap_file = os.path.join(smotif_output_dir, f"rmsd_heatmap.png")
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save RMSD matrix
        matrix_file = os.path.join(smotif_output_dir, f"rmsd_matrix.csv")
        pd.DataFrame(rmsd_matrix, index=pdb_list, columns=pdb_list).to_csv(matrix_file)
        
        # Save redundant groups
        output_file = os.path.join(smotif_output_dir, f"redundant_groups.txt")
        with open(output_file, 'w') as f:
            f.write(f"Redundant groups for {smotif_type} (RMSD cutoff: {rmsd_cutoff})\n")
            f.write("=" * 80 + "\n\n")
            
            for i, group in enumerate(redundant_groups, 1):
                f.write(f"Group {i} ({len(group)} structures):\n")
                f.write("\n".join(group))
                f.write("\n\n")
        
        stats = {
            'total_structures': len(pdb_list),
            'redundant_groups': len(redundant_groups),
            'total_redundant': sum(len(group) for group in redundant_groups)
        }
        
        print(f"Completed {smotif_type}:")
        print(f"Found {stats['redundant_groups']} redundant groups")
        print(f"Total redundant structures: {stats['total_redundant']}\n")
        
    except Exception as e:
        print(f"Error saving results for {smotif_type}: {str(e)}")

def process_all_smotif_dirs(base_dir, output_dir, rmsd_cutoff=0.5, num_processes=None):
    """
    Process all Smotif type directories in parallel
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find all Smotif type directories
    smotif_dirs = glob.glob(os.path.join(base_dir, "*_*_*"))
    print(f"Found {len(smotif_dirs)} Smotif type directories")
    
    # Check which directories need processing
    dirs_to_process = []
    for smotif_dir in smotif_dirs:
        smotif_type = os.path.basename(smotif_dir)
        smotif_output_dir = os.path.join(output_dir, smotif_type)
        
        if files_exist(smotif_output_dir):
            print(f"Skipping {smotif_type} - already processed")
        else:
            dirs_to_process.append(smotif_dir)
    
    if not dirs_to_process:
        print("All directories have been processed")
        return
    
    print(f"\nProcessing {len(dirs_to_process)} remaining directories")
    
    # Prepare arguments for parallel processing
    if num_processes is None:
        num_processes = cpu_count() - 1
    
    args_list = [(smotif_dir, output_dir, rmsd_cutoff) for smotif_dir in dirs_to_process]
    
    # Process directories in parallel
    print(f"Starting parallel processing with {num_processes} processes")
    with Pool(processes=num_processes) as pool:
        pool.map(calculate_all_vs_all_rmsd, args_list)

if __name__ == "__main__":    
    base_directory = '/home/kalabharath/projects/dingo_fold/non_redundant_smotif_coordb/center_smotif_coors'   
    output_directory= '/home/kalabharath/projects/dingo_fold/non_redundant_smotif_coordb/non_redundant'
    rmsd_threshold = 0.5
    
    process_all_smotif_dirs(base_directory, output_directory, 
                           rmsd_cutoff=rmsd_threshold,
                           num_processes=7)