import pandas as pd
import os
import math

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

def process_dataframe(input_file, output_file, pdb_folder):
    # Read the input CSV file into a DataFrame
    df = pd.read_csv(input_file)
    # print all columns as an array
    print(df.columns.values)
    exit()
    # Create a new column for total_contacts
    df['total_contacts'] = 0
    
    # Process each row
    for idx, row in df.iterrows():
        domain = row['domain']
        print(f"Processing {domain}")
        
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
        
        # Update the DataFrame directly
        df.at[idx, 'total_contacts'] = total_contacts
    
    # Save the processed DataFrame to CSV
    # only keep these columns
    
    df.to_csv(output_file, index=False)

# Usage
input_csv = './processed_extended_smotif_database2.csv'
# input_csv = './sample_db.csv'
output_csv = 'extended_smotif_db_with_contacts.csv'
# output_csv = 'test_extended_smotif_db_with_contacts.csv'
pdb_folder = '/home/kalabharath/projects/dingo_fold/cath_db/non-redundant-data-sets/dompdb/'

process_dataframe(input_csv, output_csv, pdb_folder)