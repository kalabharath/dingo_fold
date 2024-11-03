import pandas as pd
import os
import math
import numpy as np
from numba import jit, float64, int64
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

def read_pdb(file_path, chain, start, end):
    """Read CA atoms from PDB file and return as numpy array for better performance"""
    atoms = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                atom_type = line[12:16].strip()
                if atom_type == "CA":
                    res_num = int(line[22:26])
                    if chain == line[21] and start <= res_num <= end:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        atoms.append((x, y, z))
    return np.array(atoms, dtype=np.float64)

@jit(float64(float64[:], float64[:]), nopython=True)
def calculate_distance(atom1, atom2):
    """Calculate Euclidean distance between two atoms using Numba"""
    return math.sqrt(((atom1[0] - atom2[0])**2 + 
                     (atom1[1] - atom2[1])**2 + 
                     (atom1[2] - atom2[2])**2))

@jit(int64(float64[:, :], float64[:, :], float64), nopython=True)
def count_contacts_fast(sse1_atoms, sse2_atoms, cutoff):
    """Count contacts between two sets of atoms using Numba"""
    contacts = 0
    for i in range(sse1_atoms.shape[0]):
        for j in range(sse2_atoms.shape[0]):
            if calculate_distance(sse1_atoms[i], sse2_atoms[j]) <= cutoff:
                contacts += 1
    return contacts

def process_row(row_data, pdb_folder):
    """Process a single row of the DataFrame"""
    idx, row = row_data
    
    try:
        domain = row['domain']
        pdb_file = os.path.join(pdb_folder, f"{domain[:7]}.pdb")
        
        # Get atom coordinates as numpy arrays
        sse1_atoms = read_pdb(pdb_file, 
                            domain[4], 
                            int(row['SSE1_start']), 
                            int(row['SSE1_end']))
        
        sse2_atoms = read_pdb(pdb_file, 
                            domain[4], 
                            int(row['SSE2_start']), 
                            int(row['SSE2_end']))
        
        if len(sse1_atoms) == 0 or len(sse2_atoms) == 0:
            print(f"Warning: No atoms found for domain {domain}")
            return idx, 0
        
        # Calculate contacts using the Numba-optimized function
        total_contacts = count_contacts_fast(sse1_atoms, sse2_atoms, 8.0)
        
        return idx, total_contacts
    
    except Exception as e:
        print(f"Error processing domain {domain}: {str(e)}")
        return idx, 0

def process_dataframe_parallel(input_file, output_file, pdb_folder, n_processors=None, batch_size=50):
    """Process the DataFrame using parallel processing with Numba optimization"""
    
    # Read the input CSV file into a DataFrame
    df = pd.read_csv(input_file)
    print("Columns in input file:", df.columns.values)
    
    # Create a new column for total_contacts
    df['total_contacts'] = 0
    
    # Determine number of processors to use
    if n_processors is None:
        n_processors = cpu_count()
    
    # Create a pool of workers
    pool = Pool(processes=n_processors)
    
    # Prepare the rows for parallel processing
    rows_to_process = list(df.iterrows())
    
    # Create a partial function with fixed pdb_folder
    process_func = partial(process_row, pdb_folder=pdb_folder)
    
    # Process rows in parallel with progress bar
    print(f"\nProcessing {len(rows_to_process)} rows using {n_processors} processors...")
    
    # Process in batches for better memory management
    results = []
    for i in range(0, len(rows_to_process), batch_size):
        batch = rows_to_process[i:i + batch_size]
        batch_results = list(tqdm(
            pool.imap(process_func, batch),
            total=len(batch),
            desc=f"Processing batch {i//batch_size + 1}/{(len(rows_to_process) + batch_size - 1)//batch_size}"
        ))
        results.extend(batch_results)
    
    # Close the pool
    pool.close()
    pool.join()
    
    # Update the DataFrame with results
    for idx, total_contacts in results:
        df.at[idx, 'total_contacts'] = total_contacts
    
    # Keep only the specified columns
    columns_to_keep = [
        'smotif_id', 'smotif_type', 'domain', 'sse1_type', 'sse2_type', 
        'sse1_seq', 'sse2_seq', 'sse1_length', 'sse2_length', 
        'SSE1_start', 'SSE1_end', 'SSE2_start', 'SSE2_end', 
        'loop_length', 'com_distance', 'com_bin', 'total_contacts'
    ]
    df = df[columns_to_keep]
    
    # Save the processed DataFrame to CSV
    df.to_csv(output_file, index=False)
    print(f"\nProcessing complete! Results saved to {output_file}")
    
    # Print some statistics
    contacts_stats = df['total_contacts'].describe()
    print("\nContact statistics:")
    print(contacts_stats)

if __name__ == '__main__':
    # Usage
    input_csv = './processed_extended_smotif_database.csv'
    output_csv = 'extended_smotif_db_with_contacts.csv'
    pdb_folder = 'pdb_files'
    
    # Process the DataFrame in parallel with Numba optimization
    process_dataframe_parallel(
        input_csv,
        output_csv,
        pdb_folder,
        n_processors=None,  # Will use all available CPU cores
        batch_size=256  # Process in batches of 50 rows
    )