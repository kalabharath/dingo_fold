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
    try:
        atoms = []
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDB file not found: {file_path}")
            
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith("ATOM"):
                    atom_type = line[12:16].strip()
                    if atom_type == "CA":
                        try:
                            res_num = int(line[22:26].strip())
                            if chain == line[21] and start <= res_num <= end:
                                x = float(line[30:38].strip())
                                y = float(line[38:46].strip())
                                z = float(line[46:54].strip())
                                atoms.append((x, y, z))
                        except (ValueError, IndexError) as e:
                            print(f"Warning: Could not parse line in {file_path}: {line.strip()}")
                            continue
                            
        if not atoms:
            print(f"Warning: No CA atoms found in {file_path} for chain {chain} between residues {start}-{end}")
            return np.empty((0, 3), dtype=np.float64)
            
        return np.array(atoms, dtype=np.float64)
        
    except Exception as e:
        print(f"Error reading PDB file {file_path}: {str(e)}")
        return np.empty((0, 3), dtype=np.float64)
    
    
def test_batch_sizes(input_file, pdb_folder, test_sizes=[10, 50, 100, 200, 500], test_rows=1000):
    """Test different batch sizes to find optimal performance"""
    print(f"Testing batch sizes with {test_rows} rows...")
    
    # Read first test_rows rows
    df = pd.read_csv(input_file, nrows=test_rows)
    rows_to_process = list(df.iterrows())
    
    results = []
    n_processors = max(1, cpu_count() - 1)
    
    for batch_size in test_sizes:
        start_time = time.time()
        
        with Pool(processes=n_processors) as pool:
            process_func = partial(process_row, pdb_folder=pdb_folder)
            
            batch_results = []
            for i in range(0, len(rows_to_process), batch_size):
                batch = rows_to_process[i:i + batch_size]
                batch_results.extend(list(pool.imap(process_func, batch)))
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        results.append({
            'batch_size': batch_size,
            'time': elapsed_time,
            'rows_per_second': test_rows / elapsed_time
        })
        
        print(f"Batch size {batch_size}: {elapsed_time:.2f} seconds ({test_rows/elapsed_time:.2f} rows/sec)")
    
    # Find optimal batch size
    optimal = max(results, key=lambda x: x['rows_per_second'])
    print(f"\nOptimal batch size: {optimal['batch_size']}")
    print(f"Processing speed: {optimal['rows_per_second']:.2f} rows/second")
    
    return optimal['batch_size']


@jit(float64(float64[:], float64[:]), nopython=True, cache=True)
def calculate_distance(atom1, atom2):
    """Calculate Euclidean distance between two atoms using Numba"""
    return math.sqrt(((atom1[0] - atom2[0])**2 + 
                     (atom1[1] - atom2[1])**2 + 
                     (atom1[2] - atom2[2])**2))

@jit(int64(float64[:, :], float64[:, :], float64), nopython=True, cache=True)
def count_contacts_fast(sse1_atoms, sse2_atoms, cutoff):
    """Count contacts between two sets of atoms using Numba"""
    contacts = 0
    for i in range(sse1_atoms.shape[0]):
        for j in range(sse2_atoms.shape[0]):
            if calculate_distance(sse1_atoms[i], sse2_atoms[j]) <= cutoff:
                contacts += 1
    return contacts

def process_single_domain(domain, pdb_file, sse1_start, sse1_end, sse2_start, sse2_end, chain):
    """Process a single domain and return its contact count"""
    try:
        # Get atom coordinates as numpy arrays
        sse1_atoms = read_pdb(pdb_file, chain, sse1_start, sse1_end)
        sse2_atoms = read_pdb(pdb_file, chain, sse2_start, sse2_end)
        
        if len(sse1_atoms) == 0 or len(sse2_atoms) == 0:
            raise ValueError(f"No atoms found for one or both SSEs in domain {domain}")
            
        # Calculate contacts using the Numba-optimized function
        return count_contacts_fast(sse1_atoms, sse2_atoms, 8.0)
        
    except Exception as e:
        raise Exception(f"Error processing domain {domain}: {str(e)}")

def process_row(row_data, pdb_folder):
    """Process a single row of the DataFrame"""
    idx, row = row_data
    
    try:
        domain = row['domain']
        pdb_file = os.path.join(pdb_folder, f"{domain[:7]}.pdb")
        
        total_contacts = process_single_domain(
            domain=domain,
            pdb_file=pdb_file,
            sse1_start=int(row['SSE1_start']),
            sse1_end=int(row['SSE1_end']),
            sse2_start=int(row['SSE2_start']),
            sse2_end=int(row['SSE2_end']),
            chain=domain[4]
        )
        
        return idx, total_contacts
        
    except Exception as e:
        print(f"Error in row {idx}: {str(e)}")
        return idx, 0

def process_dataframe_parallel(input_file, output_file, pdb_folder, n_processors=None, batch_size=50):
    """Process the DataFrame using parallel processing with Numba optimization"""
    try:
        # Verify input file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
            
        # Verify PDB folder exists
        if not os.path.exists(pdb_folder):
            raise FileNotFoundError(f"PDB folder not found: {pdb_folder}")
        
        # Read the input CSV file into a DataFrame
        df = pd.read_csv(input_file)
        print("Columns in input file:", df.columns.values)
        
        # Verify required columns exist
        required_columns = ['domain', 'SSE1_start', 'SSE1_end', 'SSE2_start', 'SSE2_end']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Create a new column for total_contacts
        df['total_contacts'] = 0
        
        # Determine number of processors to use
        if n_processors is None:
            n_processors = max(1, cpu_count() - 1)  # Leave one CPU free
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Prepare the rows for parallel processing
        rows_to_process = list(df.iterrows())
        
        # Create a partial function with fixed pdb_folder
        process_func = partial(process_row, pdb_folder=pdb_folder)
        
        # Process rows in parallel with progress bar
        print(f"\nProcessing {len(rows_to_process)} rows using {n_processors} processors...")
        
        results = []
        with Pool(processes=n_processors) as pool:
            # Process in batches for better memory management
            for i in range(0, len(rows_to_process), batch_size):
                batch = rows_to_process[i:i + batch_size]
                batch_results = list(tqdm(
                    pool.imap(process_func, batch),
                    total=len(batch),
                    desc=f"Batch {i//batch_size + 1}/{(len(rows_to_process) + batch_size - 1)//batch_size}"
                ))
                results.extend(batch_results)
        
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
        
        # Verify all columns exist before filtering
        missing_columns = [col for col in columns_to_keep if col not in df.columns]
        if missing_columns:
            print(f"Warning: Some columns not found in DataFrame: {missing_columns}")
            columns_to_keep = [col for col in columns_to_keep if col in df.columns]
        
        df = df[columns_to_keep]
        
        # Save the processed DataFrame to CSV
        df.to_csv(output_file, index=False)
        print(f"\nProcessing complete! Results saved to {output_file}")
        
        # Print statistics
        contacts_stats = df['total_contacts'].describe()
        print("\nContact statistics:")
        print(contacts_stats)
        
        # Print additional information
        print(f"\nTotal rows processed: {len(df)}")
        print(f"Rows with zero contacts: {len(df[df['total_contacts'] == 0])}")
        
        return df
        
    except Exception as e:
        print(f"Error in main processing: {str(e)}")
        raise

if __name__ == '__main__':
    try:
        # Usage
        input_csv = './processed_extended_smotif_database.csv'
        output_csv = 'extended_smotif_db_with_contacts.csv'
        pdb_folder = 'pdb_files'
        
        # Process the DataFrame in parallel with Numba optimization
        df_result = process_dataframe_parallel(
            input_csv,
            output_csv,
            pdb_folder,
            n_processors=None,  # Will use all available CPU cores minus one
            batch_size=256  # Process in batches of 50 rows
        )
        
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise