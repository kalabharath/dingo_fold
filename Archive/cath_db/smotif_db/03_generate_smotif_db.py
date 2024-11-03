import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import logging
from datetime import datetime
# Set up logging
log_filename = f"building_smotif_db_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# Dictionary to convert three-letter amino acid codes to one-letter codes
aa_dict = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
    'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def read_pdb(file_path, chain_id):
    with open(file_path, 'r') as file:
        atoms = []
        for line in file:
            if line.startswith("ATOM") and line[21] == chain_id:
                atom_type = line[12:16].strip()
                if atom_type == "CA":
                    residue_number = int(line[22:26])
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    residue_name = line[17:20].strip()
                    atoms.append((residue_number, residue_name, np.array([x, y, z])))
    return atoms

def calculate_smotif_geometry(atoms, sse1_start, sse1_end, sse2_start, sse2_end):
    sse1_ca = [atom[2] for atom in atoms if sse1_start <= atom[0] <= sse1_end]
    sse2_ca = [atom[2] for atom in atoms if sse2_start <= atom[0] <= sse2_end]
    
    if len(sse1_ca) < 2 or len(sse2_ca) < 2:
        return None, None, None, None, None, None
    
    # Calculate D: distance between C-terminal of SS1 and N-terminal of SS2
    D = np.linalg.norm(sse2_ca[0] - sse1_ca[-1])
    
    # Calculate principal moments of inertia for each SSE
    M1 = calculate_moment_of_inertia(sse1_ca)
    M2 = calculate_moment_of_inertia(sse2_ca)
    
    # Calculate L vector
    L = sse2_ca[0] - sse1_ca[-1]
    L = L / np.linalg.norm(L)
    
    """
    # Calculate δ (hoist): angle between L and M1
    delta = np.degrees(np.arccos(np.clip(np.dot(L, M1), -1.0, 1.0)))
    
    # Calculate θ (packing): angle between M1 and M2
    theta = np.degrees(np.arccos(np.clip(np.dot(M1, M2), -1.0, 1.0)))
    """    
    # Calculate ρ (meridian): angle between M2 and plane C
    normal_P = np.cross(M1, L)
    normal_P = normal_P / np.linalg.norm(normal_P)
    
    #rho = 90 - np.degrees(np.arccos(np.clip(np.dot(M2, normal_P), -1.0, 1.0)))
    #rho = np.degrees(np.arccos(np.clip(np.dot(M2, normal_P), -1.0, 1.0)))
    #rho = (rho + 360) % 360  # Ensure rho is in [0, 360)
    

    
    delta = np.degrees(np.arccos(np.clip(np.dot(L, M1), -1.0, 1.0)))
    theta = np.degrees(np.arccos(np.clip(np.dot(M1, M2), -1.0, 1.0)))
    rho = np.degrees(np.arccos(np.clip(np.dot(M2, normal_P), -1.0, 1.0)))
    rho = (rho + 360) % 360  # Ensure rho is in [0, 360)
    
    return D, delta, theta, rho, sse1_ca, sse2_ca
    

def calculate_moment_of_inertia(coords):
    coords = np.array(coords)
    centroid = np.mean(coords, axis=0)
    centered_coords = coords - centroid
    inertia_tensor = np.dot(centered_coords.T, centered_coords)
    eigenvalues, eigenvectors = np.linalg.eig(inertia_tensor)
    return eigenvectors[:, np.argmax(eigenvalues)]

def bin_parametersOLD(D, delta, theta, rho):
    D_bin = min(int(D / 4), 9)
    delta_bin = int(delta / 60)
    theta_bin = int(theta / 60)
    rho_bin = int((rho + 180) / 60) % 6  # Adjust rho to be in [0, 360)
    return D_bin, delta_bin, theta_bin, rho_bin

def bin_parametersOLD(D, delta, theta, rho):
    D_bin = min(int(D / 4), 9)
    delta_bin = min(int(delta / 60), 2)
    theta_bin = min(int(theta / 60), 2)
    rho_bin = int(rho / 60) % 6
    return D_bin, delta_bin, theta_bin, rho_bin

def bin_parameters(D, delta, theta, rho):
    D_bin = min(int(D / 4), 9)
    delta_bin = min(int(delta / 60), 2)
    theta_bin = min(int(theta / 60), 2)
    rho_bin = int(rho / 60)
    return D_bin, delta_bin, theta_bin, rho_bin

def extract_sequence(atoms, start, end):
    return ''.join([aa_dict.get(atom[1], 'X') for atom in atoms if start <= atom[0] <= end])

def process_smotif(pdb_file, row):
    try:
        chain_id = row['Domain'][4]  # Assuming the 5th character is the chain ID
        atoms = read_pdb(pdb_file, chain_id)
        
        geom_results = calculate_smotif_geometry(
            atoms,
            row['SSE1_Start'], row['SSE1_End'],
            row['SSE2_Start'], row['SSE2_End']
        )
        
        if geom_results[0] is None:
            logger.warning(f"Skipping Smotif in {row['Domain']} due to geometry calculation failure")
            return None
        
        D, delta, theta, rho, sse1_coords, sse2_coords = geom_results
        D_bin, delta_bin, theta_bin, rho_bin = bin_parameters(D, delta, theta, rho)
        
        sse1_seq = extract_sequence(atoms, row['SSE1_Start'], row['SSE1_End'])
        sse2_seq = extract_sequence(atoms, row['SSE2_Start'], row['SSE2_End'])
        loop_seq = extract_sequence(atoms, row['SSE1_End']+1, row['SSE2_Start']-1)
        
        #smotif_id = f"{row['SSE1_Type']}{row['SSE2_Type']}_{D_bin}_{delta_bin}_{theta_bin}_{rho_bin}"
        
        
        smotif_type = f"{row['SSE1_Type']}{row['SSE2_Type']}"
        smotif_id = f"{smotif_type}_{D_bin}_{delta_bin}_{theta_bin}_{rho_bin}"
    
    
        return {
            'smotif_id': smotif_id,
            'smotif_type': smotif_type,
            'domain': row['Domain'],
            'sse1_type': row['SSE1_Type'],
            'sse2_type': row['SSE2_Type'],
            'sse1_seq': sse1_seq,
            'sse2_seq': sse2_seq,
            'loop_seq': loop_seq,
            'sse1_length': row['SSE1_Length'],
            'sse2_length': row['SSE2_Length'],
            'loop_length': row['Loop_Length'],
            'D': D,
            'delta': delta,
            'theta': theta,
            'rho': rho,
            'D_bin': D_bin,
            'delta_bin': delta_bin,
            'theta_bin': theta_bin,
            'rho_bin': rho_bin            
        }
    except Exception as e:
        logger.error(f"Error processing Smotif in {row['Domain']}: {str(e)}")
        return None

def create_smotif_database(smotif_csv, pdb_dir):
    df = pd.read_csv(smotif_csv)
    smotif_data = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Smotifs"):
        pdb_file = os.path.join(pdb_dir, f"{row['Domain'][:7]}.pdb")
        if not os.path.exists(pdb_file):
            logger.warning(f"PDB file not found: {pdb_file}")
            continue
        
        smotif = process_smotif(pdb_file, row)
        if smotif is not None:
            smotif_data.append(smotif)
    
    return pd.DataFrame(smotif_data)

# Usage
smotif_csv = 'sample_smottif_db.csv'
smotif_csv = 'smotif_database2.csv'
pdb_dir = '/home/kalabharath/projects/dingo_fold/cath_db/non-redundant-data-sets/dompdb'
output_file = 'processed_smotif_database2.csv'

logger.info("Starting Smotif database creation")
smotif_db = create_smotif_database(smotif_csv, pdb_dir)
logger.info(f"Created Smotif database with {len(smotif_db)} entries")

# Save the database
smotif_db.to_csv(output_file, index=False)
logger.info(f"Saved processed Smotif database to {output_file}")

# Print some statistics
logger.info(f"Total Smotifs: {len(smotif_db)}")
logger.info(f"Unique Smotif types: {smotif_db['smotif_id'].nunique()}")
logger.info(f"Average loop length: {smotif_db['loop_length'].mean():.2f}")
logger.info("Smotif type distribution:")
logger.info(smotif_db['sse1_type'] + smotif_db['sse2_type'].value_counts(normalize=True))

# Print detailed statistics
logger.info("Smotif type distribution:")
logger.info(smotif_db['smotif_type'].value_counts())

for smotif_type in ['HH', 'HE', 'EH', 'EE']:
    type_count = smotif_db[smotif_db['smotif_type'] == smotif_type]['smotif_id'].nunique()
    logger.info(f"Unique {smotif_type} Smotif types: {type_count}")

# Check for missing Smotif types
all_possible_ids = [f"{st}_{d}_{de}_{t}_{r}" for st in ['HH', 'HE', 'EH', 'EE'] 
                    for d in range(10) for de in range(3) for t in range(3) for r in range(6)]
missing_ids = set(all_possible_ids) - set(smotif_db['smotif_id'])
logger.info(f"Number of missing Smotif types: {len(missing_ids)}")