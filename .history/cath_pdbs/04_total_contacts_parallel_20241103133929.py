import pandas as pd
import os
import math
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

def read_pdb(file_path, chain, start, end):
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
    return atoms

def calculate_distance(atom1, atom2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(atom1, atom2)))

def count_contacts(sse1_atoms, sse2_atoms, cutoff=8.0):
    contacts = 0
    for atom1 in sse1_atoms:
        for atom2 in sse2_atoms:
            if calculate_distance(atom1, atom2) <= cutoff:
                contacts += 1
    return contacts

def process_row(row_data, pdb_folder):
    """Process a single row of the DataFrame"""
    idx, row = row_data
    
    try:
        domain = row['domain']
        pdb_file = os.path.join(pdb_folder, f"{domain[:7]}.pdb")
        
        sse1_atoms = read_pdb(pdb_file, 
                            domain[4], 
                            int(row['SSE1_start']), 
                            int(row['SSE1_end']))
        
        sse2_atoms = read_pdb(pdb_file, 
                            domain[4], 
                            int(row['SSE2_start']), 
                            int(row['SSE2_end']))
        
        total_contacts = count_contacts(sse1_atoms, sse2_atoms)
        
        return idx, total_contacts
    
    except Exception as e:
        print(f"Error processing domain {domain}: {str(e)}")
        return idx, 0

def process_dataframe_parallel(input_file, output_file, pdb_folder, n_processors=None):
    """Process the DataFrame using parallel processing"""
    
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
    results = list(tqdm(
        pool.imap(process_func, rows_to_process),
        total=len(rows_to_process),
        desc="Processing domains"
    ))
    
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
    
    # Process the DataFrame in parallel
    process_dataframe_parallel(
        input_csv,
        output_csv,
        pdb_folder,
        n_processors=None  # Will use all available CPU cores
    )