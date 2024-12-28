import pandas as pd
import os
import glob

def extract_smotif_coordinates(csv_file, pdb_dir, output_dir):
    """
    Extract Smotif coordinates from PDB files based on CSV definitions
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    for _, row in df.iterrows():
        try:
            # Get domain name (PDB file name)
            pdb_name = row['domain']
            # remove '.stride' from the pdb name
            pdb_name = pdb_name.replace('.stride', '')
            
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
            
            # Write extracted coordinates to new PDB file
            output_file = os.path.join(output_dir, f"{row['smotif_id']}.pdb")
            with open(output_file, 'w') as f:
                f.write("HEADER    SMOTIF COORDINATES\n")
                for line in smotif_lines:
                    f.write(line)
                f.write("END\n")
                
        except Exception as e:
            print(f"Error processing {row['smotif_id']}: {str(e)}")
            continue

# Usage
if __name__ == "__main__":
    csv_directory = '/home/kalabharath/projects/dingo_fold/cath_pdbs/group_smotifs'
    pdb_directory = '/home/kalabharath/projects/dingo_fold/cath_pdbs/pdb_files'
    output_directory = '/home/kalabharath/projects/dingo_fold/non_redundant_smotif_coordb/smotif_coors'
    
    # Process each CSV file
    for csv_file in glob.glob(os.path.join(csv_directory, "*.csv")):
        extract_smotif_coordinates(csv_file, pdb_directory, output_directory)