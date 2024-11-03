import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import logging
from collections import Counter
from datetime import datetime
import itertools

# Set up logging
log_filename = f"building_extended_smotif_db_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

def calculate_center_of_mass(coords):
    return np.mean(coords, axis=0)

def calculate_smotif_geometry(atoms, sse1_start, sse1_end, sse2_start, sse2_end):
    sse1_ca = [atom[2] for atom in atoms if sse1_start <= atom[0] <= sse1_end]
    sse2_ca = [atom[2] for atom in atoms if sse2_start <= atom[0] <= sse2_end]
    
    if len(sse1_ca) < 2 or len(sse2_ca) < 2:
        return None, None, None, None, None, None, None
    
    # Calculate D: distance between C-terminal of SS1 and N-terminal of SS2
    D = np.linalg.norm(sse2_ca[0] - sse1_ca[-1])
    
    # Calculate center of mass distance
    com1 = calculate_center_of_mass(sse1_ca)
    com2 = calculate_center_of_mass(sse2_ca)
    com_distance = np.linalg.norm(com2 - com1)
    
    # Calculate principal moments of inertia for each SSE
    M1 = calculate_moment_of_inertia(sse1_ca)
    M2 = calculate_moment_of_inertia(sse2_ca)
    
    # Calculate L vector
    L = sse2_ca[0] - sse1_ca[-1]
    L = L / np.linalg.norm(L)
    
    # Calculate δ (hoist): angle between L and M1
    delta = np.degrees(np.arccos(np.clip(np.dot(L, M1), -1.0, 1.0)))
    
    # Calculate θ (packing): angle between M1 and M2
    theta = np.degrees(np.arccos(np.clip(np.dot(M1, M2), -1.0, 1.0)))
    
    # Calculate ρ (meridian): angle between M2 and plane C
    normal_P = np.cross(M1, L)
    normal_P = normal_P / np.linalg.norm(normal_P)
        
    # Calculate plane C (perpendicular to both M1 and normal_P)
    normal_C = np.cross(M1, normal_P)
    normal_C = normal_C / np.linalg.norm(normal_C)

    # Calculate ρ (meridian): angle between M2 and plane C    
    rho = (90 - np.degrees(np.arccos(np.clip(np.dot(M2, normal_P), -1.0, 1.0)))) % 360
    
    return D, delta, theta, rho, com_distance, sse1_ca, sse2_ca

def calculate_moment_of_inertia(coords):
    coords = np.array(coords)
    centroid = np.mean(coords, axis=0)
    centered_coords = coords - centroid
    inertia_tensor = np.dot(centered_coords.T, centered_coords)
    eigenvalues, eigenvectors = np.linalg.eig(inertia_tensor)
    return eigenvectors[:, np.argmax(eigenvalues)]

def bin_parameters(D, delta, theta, rho, com_distance):
    # Distance binning
    if D <= 10:
        D_bin = int(D)
    elif D <= 20:
        D_bin = 10 + int((D - 10) / 2)
    elif D <= 40:
        D_bin = 15 + int((D - 20) / 5)
    else:
        D_bin = 19

    # Hoist angle binning
    if delta <= 90:
        delta_bin = int(delta / 15)
    else:
        delta_bin = 6 + int((delta - 90) / 30)

    # Packing angle binning
    if theta <= 90:
        theta_bin = int(theta / 15)
    else:
        theta_bin = 6 + int((theta - 90) / 30)

    # Meridian angle binning
    rho_bin = int(rho / 30)
    
    # COM distance binning
    if com_distance <= 20:
        com_bin = int(com_distance)
    elif com_distance <= 40:
        com_bin = 20 + int((com_distance - 20) / 2)
    else:
        com_bin = 0

    return D_bin, delta_bin, theta_bin, rho_bin, com_bin

def extract_sequence(atoms, start, end):
    return ''.join([aa_dict.get(atom[1], 'X') for atom in atoms if start <= atom[0] <= end])

def process_extended_smotif(pdb_file, sse1, sse2):
    try:
        chain_id = sse1['Domain'][4]  # Assuming the 5th character is the chain ID
        atoms = read_pdb(pdb_file, chain_id)
        
        geom_results = calculate_smotif_geometry(
            atoms,
            sse1['SSE1_Start'], sse1['SSE1_End'],
            sse2['SSE1_Start'], sse2['SSE1_End']
        )
        
        if geom_results[0] is None:
            logger.warning(f"Skipping extended Smotif in {sse1['Domain']} due to geometry calculation failure")
            return None
        
        D, delta, theta, rho, com_distance, sse1_coords, sse2_coords = geom_results
        D_bin, delta_bin, theta_bin, rho_bin, com_bin = bin_parameters(D, delta, theta, rho, com_distance)
        loop_length = sse2['SSE1_Start'] - sse1['SSE1_End'] - 1
        
        logger.debug(f"Extended Smotif in {sse1['Domain']}: D={D:.2f}, delta={delta:.2f}, theta={theta:.2f}, rho={rho:.2f}, COM_distance={com_distance:.2f}")
        logger.debug(f"Binned: D_bin={D_bin}, delta_bin={delta_bin}, theta_bin={theta_bin}, rho_bin={rho_bin}, com_bin={com_bin}")
        
        sse1_seq = extract_sequence(atoms, sse1['SSE1_Start'], sse1['SSE1_End'])
        sse2_seq = extract_sequence(atoms, sse2['SSE1_Start'], sse2['SSE1_End'])
        
        smotif_type = f"{sse1['SSE1_Type']}{sse2['SSE1_Type']}"
        smotif_id = f"{smotif_type}_{D_bin:02d}_{delta_bin:02d}_{theta_bin:02d}_{rho_bin:02d}"
        smotif_id = f"{smotif_type}_{sse1['SSE1_Length']}_{sse2['SSE1_Length'],}_{com_distance:02d}_{loop_length:02d}"
        # add loop length to smotif_id
        
        return {
            'smotif_id': smotif_id,
            'smotif_type': smotif_type,
            'domain': sse1['Domain'],
            'sse1_type': sse1['SSE1_Type'],
            'sse2_type': sse2['SSE1_Type'],
            'sse1_seq': sse1_seq,
            'sse2_seq': sse2_seq,
            'sse1_length': sse1['SSE1_Length'],
            'sse2_length': sse2['SSE1_Length'],
            'SSE1_start': sse1['SSE1_Start'],
            'SSE1_end': sse1['SSE1_End'],
            'SSE2_start': sse2['SSE1_Start'],
            'SSE2_end': sse2['SSE1_End'],
            'loop_length': loop_length, 
            'com_distance': com_distance,
            'D': D,
            'delta': delta,
            'theta': theta,
            'rho': rho,            
            'D_bin': D_bin,
            'delta_bin': delta_bin,
            'theta_bin': theta_bin,
            'rho_bin': rho_bin,
            'com_bin': com_bin            
        }
    except Exception as e:
        logger.error(f"Error processing extended Smotif in {sse1['Domain']}: {str(e)}")
        return None

def create_extended_smotif_database(smotif_csv, pdb_dir):
    df = pd.read_csv(smotif_csv)
    extended_smotif_data = []
    
    for domain, group in tqdm(df.groupby('Domain'), desc="Processing Domains"):
        pdb_file = os.path.join(pdb_dir, f"{domain[:7]}.pdb")
        if not os.path.exists(pdb_file):
            logger.warning(f"PDB file not found: {pdb_file}")
            continue
        
        sses = group.to_dict('records')
        for sse1, sse2 in itertools.combinations(sses, 2):
            extended_smotif = process_extended_smotif(pdb_file, sse1, sse2)
            if extended_smotif is not None:
                extended_smotif_data.append(extended_smotif)
    
    return pd.DataFrame(extended_smotif_data)

# Usage
smotif_csv = 'smotif_database2.csv'
pdb_dir = '/home/kalabharath/projects/dingo_fold/cath_db/non-redundant-data-sets/dompdb'
output_file = 'processed_extended_smotif_database2.csv'

logger.info("Starting extended Smotif database creation")
extended_smotif_db = create_extended_smotif_database(smotif_csv, pdb_dir)
logger.info(f"Created extended Smotif database with {len(extended_smotif_db)} entries")

# Save the database
extended_smotif_db.to_csv(output_file, index=False)
logger.info(f"Saved processed extended Smotif database to {output_file}")

# Print statistics
logger.info(f"Total extended Smotifs: {len(extended_smotif_db)}")
logger.info(f"Unique extended Smotif types: {extended_smotif_db['smotif_id'].nunique()}")
logger.info("Extended Smotif type distribution:")
logger.info(extended_smotif_db['smotif_type'].value_counts())

for smotif_type in ['HH', 'HE', 'EH', 'EE']:
    type_count = extended_smotif_db[extended_smotif_db['smotif_type'] == smotif_type]['smotif_id'].nunique()
    logger.info(f"Unique {smotif_type} extended Smotif types: {type_count}")

# Analyze missing extended Smotif types
all_possible_ids = [f"{st}_{d:02d}_{de:02d}_{t:02d}_{r:02d}_{c:02d}" 
                    for st in ['HH', 'HE', 'EH', 'EE'] 
                    for d in range(20) for de in range(9) 
                    for t in range(9) for r in range(12) for c in range(31)]

existing_ids = set(extended_smotif_db['smotif_id'])
missing_ids = set(all_possible_ids) - existing_ids

logger.info(f"Total possible extended Smotif types: {len(all_possible_ids)}")
logger.info(f"Observed extended Smotif types: {len(existing_ids)}")
logger.info(f"Missing extended Smotif types: {len(missing_ids)}")

# Analyze distribution of missing types
missing_types = [id.split('_')[0] for id in missing_ids]
for st in ['HH', 'HE', 'EH', 'EE']:
    count = missing_types.count(st)
    logger.info(f"Missing {st} types: {count}")

# Analyze distribution of geometric bins in missing types
for i, param in enumerate(['D', 'delta', 'theta', 'rho', 'com']):
    bins = [int(id.split('_')[i+1]) for id in missing_ids]
    logger.info(f"Distribution of missing {param} bins: {Counter(bins)}")