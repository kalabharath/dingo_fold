import pandas as pd
import os
import glob

def extract_smotif_coordinates(csv_file, pdb_dir, output_dir):
    """
    Extract Smotif coordinates from PDB files based on CSV definitions
    """
    # Create output directory named after the CSV file
    csv_name = os.path.splitext(os.path.basename(csv_file))[0]
    csv_output_dir = os.path.join(output_dir, csv_name)
    if not os.path.exists(csv_output_dir):
        os.makedirs(csv_output_dir)
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    print(f"Processing {csv_file}...")
    
    processed = 0
    for _, row in df.iterrows():
        try:
            # Get domain name (PDB file name)
            pdb_name = row['domain']
            pdb_file = os.path.join(pdb_dir, pdb_name)
            
            # Get Smotif residue ranges
            sse1_start = int(row['SSE1_start'])
            sse1_end = int(row['SSE1_end'])
            sse2_start = int(row['SSE2_start'])
            sse2_end = int(row['SSE2_end'])
            
            # Read PDB file and extract relevant coordinates
            smotif_lines = []
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith('ATOM  ') or line.startswith('HETATM'):
                        res_num = int(line[22:26])
                        if (sse1_start <= res_num <= sse1_end) or (sse2_start <= res_num <= sse2_end):
                            smotif_lines.append(line)
            
            # Create output filename with both smotif_id and pdb_name
            output_file = os.path.join(csv_output_dir, f"{row['smotif_id']}_{pdb_name}")
            with open(output_file, 'w') as f:
                f.write(f"HEADER    SMOTIF COORDINATES FROM {pdb_name}\n")
                f.write(f"REMARK    SSE1: {sse1_start}-{sse1_end}\n")
                f.write(f"REMARK    SSE2: {sse2_start}-{sse2_end}\n")
                for line in smotif_lines:
                    f.write(line)
                f.write("END\n")
            
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed} entries...")
                
        except Exception as e:
            print(f"Error processing {row['smotif_id']} from {pdb_name}: {str(e)}")
            continue
    
    print(f"Finished processing {csv_file}. Processed {processed} entries.")

# Usage
if __name__ == "__main__":
    
    # Process each CSV file
    csv_files = glob.glob(os.path.join(csv_directory, "*.csv"))
    print(f"Found {len(csv_files)} CSV files to process")
    
    for csv_file in csv_files:
        extract_smotif_coordinates(csv_file, pdb_directory, output_directory)