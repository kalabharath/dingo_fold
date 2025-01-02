import numpy as np
import os
import glob
from nqcp import centerCoo, getCAcoo
from qcp_numba import CalcRMSDRotationalMatrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_rmsd_heatmap(rmsd_matrix, labels, output_file):
    """
    Create and save a heatmap visualization of RMSD matrix
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(rmsd_matrix, 
                xticklabels=labels, 
                yticklabels=labels,
                cmap='viridis',
                square=True)
    plt.title('RMSD Matrix')
    plt.xlabel('Structure')
    plt.ylabel('Structure')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_all_vs_all_rmsd(smotif_dir, output_dir, rmsd_cutoff=0.5):
    """
    Calculate all-vs-all RMSD for PDB files in a directory
    """
    # Get all PDB files
    pdb_files = glob.glob(os.path.join(smotif_dir, "*.pdb"))
    print(f"Processing {len(pdb_files)} structures in {smotif_dir}")
    
    # Store coordinates for each structure
    structures = {}
    for pdb_file in pdb_files:
        try:
            coords = []
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith(('ATOM', 'HETATM')):
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords.append([x, y, z])
            
            coords = np.array(coords).T  # Convert to 3xN format
            coords, _ = centerCoo(coords)  # Center coordinates
            structures[os.path.basename(pdb_file)] = coords
            
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
        coords1 = structures[pdb1]
        
        for j, pdb2 in enumerate(pdb_list[i:], i):
            coords2 = structures[pdb2]
            
            # Skip if different number of atoms
            if coords1.shape != coords2.shape:
                rmsd_matrix[i,j] = rmsd_matrix[j,i] = np.nan
                continue
                
            try:
                rotation_matrix, rmsd = CalcRMSDRotationalMatrix(coords1, coords2, len(coords1[0]))
                rmsd_matrix[i,j] = rmsd_matrix[j,i] = rmsd
                
                # Collect redundant groups (only process upper triangle)
                if i != j and rmsd <= rmsd_cutoff and pdb1 not in processed:
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
            print(f"Processed {i+1}/{n_structs} structures")
    
    # Create heatmap
    smotif_type = os.path.basename(smotif_dir)
    heatmap_file = os.path.join(output_dir, f"{smotif_type}_rmsd_heatmap.png")
    create_rmsd_heatmap(rmsd_matrix, pdb_list, heatmap_file)
    
    return redundant_groups, rmsd_matrix, pdb_list

def process_all_smotif_dirs(base_dir, output_dir, rmsd_cutoff=0.5):
    """
    Process all Smotif type directories and save results
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Find all Smotif type directories
    smotif_dirs = glob.glob(os.path.join(base_dir, "*_*_*"))
    print(f"Found {len(smotif_dirs)} Smotif type directories")
    
    for smotif_dir in smotif_dirs:
        smotif_type = os.path.basename(smotif_dir)
        print(f"\nProcessing {smotif_type}")
        
        try:
            redundant_groups, rmsd_matrix, pdb_list = calculate_all_vs_all_rmsd(
                smotif_dir, output_dir, rmsd_cutoff
            )
            
            # Save RMSD matrix as CSV
            matrix_file = os.path.join(output_dir, f"{smotif_type}_rmsd_matrix.csv")
            pd.DataFrame(rmsd_matrix, index=pdb_list, columns=pdb_list).to_csv(matrix_file)
            
            # Save redundant groups
            output_file = os.path.join(output_dir, f"{smotif_type}_redundant_groups.txt")
            with open(output_file, 'w') as f:
                f.write(f"Redundant groups for {smotif_type} (RMSD cutoff: {rmsd_cutoff})\n")
                f.write("=" * 80 + "\n\n")
                
                for i, group in enumerate(redundant_groups, 1):
                    f.write(f"Group {i} ({len(group)} structures):\n")
                    f.write("\n".join(group))
                    f.write("\n\n")
            
            # Save summary statistics
            stats = {
                'total_structures': len(pdb_list),
                'redundant_groups': len(redundant_groups),
                'total_redundant': sum(len(group) for group in redundant_groups)
            }
            
            print(f"Found {stats['redundant_groups']} redundant groups")
            print(f"Total redundant structures: {stats['total_redundant']}")
            print(f"Generated heatmap: {smotif_type}_rmsd_heatmap.png")
            
        except Exception as e:
            print(f"Error processing directory {smotif_type}: {str(e)}")
            continue

if __name__ == "__main__":
    # Replace these paths with your actual paths
    base_directory = "/path/to/your/smotif/directories"  # Directory containing HE_37_8, HH_11_29, etc.
    base_directory = '/home/kalabharath/projects/dingo_fold/non_redundant_smotif_coordb/center_smotif_coors'
    output_directory = "/path/to/output"  # Where to save results
    
    rmsd_threshold = 0.5  # Adjust this value as needed
    
    process_all_smotif_dirs(base_directory, output_directory, rmsd_threshold)