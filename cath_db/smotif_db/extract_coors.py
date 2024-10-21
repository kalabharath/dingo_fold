import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def read_pdb(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("ATOM"):
                yield line

def extract_backbone_coords(pdb_lines, chain_id, start, end):
    coords = []
    seq = []
    current_res = None
    res_atoms = {'N': None, 'CA': None, 'C': None, 'O': None}
    
    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }

    def add_residue():
        coords.extend([res_atoms['N'] or [np.nan, np.nan, np.nan],
                       res_atoms['CA'] or [np.nan, np.nan, np.nan],
                       res_atoms['C'] or [np.nan, np.nan, np.nan],
                       res_atoms['O'] or [np.nan, np.nan, np.nan]])
        seq.append(three_to_one.get(res_name, 'X'))

    for line in pdb_lines:
        if line[21] == chain_id:
            res_num = int(line[22:26])
            if start <= res_num <= end:
                atom_name = line[12:16].strip()
                if atom_name in ['N', 'CA', 'C', 'O']:
                    if res_num != current_res:
                        if current_res is not None:
                            add_residue()
                        current_res = res_num
                        res_atoms = {'N': None, 'CA': None, 'C': None, 'O': None}
                        res_name = line[17:20]
                    
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    res_atoms[atom_name] = [x, y, z]

    if current_res is not None:
        add_residue()

    return np.array(coords).reshape(-1, 3), ''.join(seq)

def write_pdb(filename, coords, seq):
    with open(filename, 'w') as f:
        atom_number = 1
        res_number = 1
        for i, aa in enumerate(seq):
            for j, atom_name in enumerate(['N', 'CA', 'C', 'O']):
                if not np.isnan(coords[i*4+j][0]):  # Check if coordinates exist
                    x, y, z = coords[i*4+j]
                    f.write(f"ATOM  {atom_number:5d} {atom_name:^4s} {aa:3s} A{res_number:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {atom_name[0]}  \n")
                    atom_number += 1
            res_number += 1
        f.write("END\n")

# Load the data
df = pd.read_csv('processed_smotif_database3.csv')

# Create a directory to store PDB files
os.makedirs('smotif_pdbs', exist_ok=True)

# Counter for each smotif_id
smotif_counters = {}

# Group by smotif_id
for smotif_id, group in tqdm(df.groupby('smotif_id'), desc="Processing Smotifs"):
    # Create a subdirectory for each Smotif type
    smotif_type = group['smotif_type'].iloc[0]
    os.makedirs(f'smotif_pdbs/{smotif_type}', exist_ok=True)
    
    # Initialize counter for this smotif_id if not exists
    if smotif_id not in smotif_counters:
        smotif_counters[smotif_id] = 0
    
    # Process each instance of this Smotif
    for _, smotif in group.iterrows():
        # Extract domain and chain information
        domain = smotif['domain']
        pdb_id = domain[:7]
        chain_id = domain[4]
        
        # Load the parent PDB file
        pdb_file = f"/home/kalabharath/projects/dingo_fold/cath_db/non-redundant-data-sets/dompdb/{pdb_id}.pdb"  # Adjust this path
        try:
            pdb_lines = list(read_pdb(pdb_file))
        except FileNotFoundError:
            print(f"PDB file not found for {pdb_id}. Skipping this Smotif.")
            continue
        
        # Extract coordinates
        sse1_coords, sse1_seq = extract_backbone_coords(pdb_lines, chain_id, smotif['SSE1_start'], smotif['SSE1_end'])
        sse2_coords, sse2_seq = extract_backbone_coords(pdb_lines, chain_id, smotif['SSE2_start'], smotif['SSE2_end'])
        loop_coords, loop_seq = extract_backbone_coords(pdb_lines, chain_id, smotif['SSE1_end']+1, smotif['SSE2_start']-1)
        
        # Combine coordinates and sequence
        coords = np.vstack((sse1_coords, loop_coords, sse2_coords)) if len(loop_coords) > 0 else np.vstack((sse1_coords, sse2_coords))
        seq = sse1_seq + loop_seq + sse2_seq
        
        # Write PDB file with unique suffix
        smotif_counters[smotif_id] += 1
        filename = f"smotif_pdbs/{smotif_type}/{smotif_id}_{smotif_counters[smotif_id]:04d}.pdb"
        write_pdb(filename, coords, seq)

print("PDB files have been generated in the 'smotif_pdbs' directory.")