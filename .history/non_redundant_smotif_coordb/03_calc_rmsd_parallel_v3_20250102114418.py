import numpy as np
import os
import glob
from numba_rmsd.nqcp import centerCoo, getCAcoo
from numba_rmsd.qcp_numba import CalcRMSDRotationalMatrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from shutil import copy2

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
    """Check if all output files exist for a given Smotif directory"""
    required_files = [
        "rmsd_heatmap.png",
        "rmsd_matrix.csv",
        "redundant_groups.txt"
    ]
    
    return all(os.path.exists(os.path.join(smotif_output_dir, f)) for f in required_files)

def create_sorted_heatmap(rmsd_matrix, pdb_list, smotif_type, smotif_output_dir):
    """Create a sorted heatmap based on RMSD values"""
    # Convert to DataFrame for easier handling
    df = pd.DataFrame(rmsd_matrix, index=pdb_list, columns=pdb_list)
    
    # Ensure diagonal is exactly zero
    np.fill_diagonal(df.values, 0.0)
    
    # Handle NaN values for clustering
    # Replace NaN with maximum finite value
    max_val = np.nanmax(df.values)
    matrix_for_clustering = df.fillna(max_val).values
    
    # Ensure matrix is symmetric
    matrix_for_clustering = (matrix_for_clustering + matrix_for_clustering.T) / 2
    
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import squareform
    
    try:
        # Convert to condensed form
        condensed_dist = squareform(matrix_for_clustering)
        
        # Perform clustering
        linkage = hierarchy.linkage(condensed_dist, method='average')
        
        # Get the order of items
        sorted_idx = hierarchy.leaves_list(linkage)
        
        # Reorder the matrix
        sorted_df = df.iloc[sorted_idx, sorted_idx]
        
    except Exception as e:
        print(f"Clustering failed for {smotif_type}, using original order: {str(e)}")
        sorted_df = df
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(sorted_df,
                xticklabels=sorted_df.columns,
                yticklabels=sorted_df.index,
                cmap='viridis',
                square=True)
    plt.title(f'RMSD Matrix - {smotif_type} (Sorted by similarity)')
    plt.xlabel('Structure')
    plt.ylabel('Structure')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the sorted heatmap
    heatmap_file = os.path.join(smotif_output_dir, f"rmsd_heatmap.png")
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save sorted RMSD matrix
    matrix_file = os.path.join(smotif_output_dir, f"rmsd_matrix.csv")
    sorted_df.to_csv(matrix_file)
    
    return sorted_df.index.tolist()  # Return sorted PDB list

def create_non_redundant_library(smotif_dir, output_dir, redundant_groups, sorted_pdb_list):
    """Create non-redundant library by copying over unique structures"""
    # Create directory for non-redundant structures
    nonred_dir = os.path.join(output_dir, "non_redundant_pdbs")
    os.makedirs(nonred_dir, exist_ok=True)
    
    # Get all redundant PDB names
    redundant_pdbs = set()
    kept_pdbs = set()
    for group in redundant_groups:
        # Keep first PDB in sorted order as representative
        kept_pdbs.add(group[0])
        # Mark rest as redundant
        redundant_pdbs.update(group[1:])
    
    # Copy over non-redundant structures
    smotif_type = os.path.basename(smotif_dir)
    copied_files = []
    
    # Create Smotif-specific directory
    smotif_nonred_dir = os.path.join(nonred_dir, smotif_type)
    os.makedirs(smotif_nonred_dir, exist_ok=True)
    
    # Follow sorted order for copying
    for pdb in sorted_pdb_list:
        if pdb not in redundant_pdbs:
            src = os.path.join(smotif_dir, pdb)
            dst = os.path.join(smotif_nonred_dir, pdb)
            try:
                copy2(src, dst)
                copied_files.append(pdb)
            except Exception as e:
                print(f"Error copying {pdb}: {str(e)}")
    
    # Save summary
    summary_file = os.path.join(smotif_nonred_dir, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Non-redundant library for {smotif_type}\n")
        f.write("=" * 80 + "\n\n")
        f.write("Statistics:\n")
        f.write(f"Total structures: {len(sorted_pdb_list)}\n")
        f.write(f"Redundant structures removed: {len(redundant_pdbs)}\n")
        f.write(f"Representatives kept: {len(kept_pdbs)}\n")
        f.write(f"Non-redundant structures: {len(copied_files)}\n\n")
        
        f.write("Kept from redundant groups:\n")
        for group in redundant_groups:
            f.write(f"Representative: {group[0]}\n")
            f.write(f"Removed: {', '.join(group[1:])}\n\n")
        
        f.write("\nAll kept structures:\n")
        f.write('\n'.join(copied_files))
    
    return len(copied_files)

def calculate_all_vs_all_rmsd(args):
    """Calculate all-vs-all RMSD for PDB files in a directory"""
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
        # Create sorted heatmap and get sorted PDB list
        sorted_pdb_list = create_sorted_heatmap(rmsd_matrix, pdb_list, smotif_type, smotif_output_dir)
        
        # Save redundant groups (using sorted order)
        output_file = os.path.join(smotif_output_dir, f"redundant_groups.txt")
        with open(output_file, 'w') as f:
            f.write(f"Redundant groups for {smotif_type} (RMSD cutoff: {rmsd_cutoff})\n")
            f.write("=" * 80 + "\n\n")
            
            # Sort redundant groups based on sorted PDB list
            sorted_groups = []
            for group in redundant_groups:
                # Sort members within group based on their order in sorted_pdb_list
                sorted_group = sorted(group, key=lambda x: sorted_pdb_list.index(x))
                sorted_groups.append(sorted_group)
            
            # Sort groups themselves based on their first member
            sorted_groups.sort(key=lambda x: sorted_pdb_list.index(x[0]))
            
            for i, group in enumerate(sorted_groups, 1):
                f.write(f"Group {i} ({len(group)} structures):\n")
                f.write("\n".join(group))
                f.write("\n\n")
        
        # Create non-redundant library
        num_kept = create_non_redundant_library(smotif_dir, output_dir, sorted_groups, sorted_pdb_list)
        
        stats = {
            'total_structures': len(pdb_list),
            'redundant_groups': len(redundant_groups),
            'total_redundant': sum(len(group) for group in redundant_groups),
            'non_redundant_kept': num_kept
        }
        
        print(f"Completed {smotif_type}:")
        print(f"Found {stats['redundant_groups']} redundant groups")
        print(f"Total redundant structures: {stats['total_redundant']}")
        print(f"Kept {stats['non_redundant_kept']} non-redundant structures\n")
        
    except Exception as e:
        print(f"Error saving results for {smotif_type}: {str(e)}")

def process_all_smotif_dirs(base_dir, output_dir, rmsd_cutoff=0.5, num_processes=None):
    """Process all Smotif type directories in parallel"""
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
    output_directory = '/home/kalabharath/projects/dingo_fold/non_redundant_smotif_coordb/non_redundant'
    rmsd_threshold = 0.5
    
    process_all_smotif_dirs(base_directory, output_directory, 
                           rmsd_cutoff=rmsd_threshold,
                           num_processes=1)