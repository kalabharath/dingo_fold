import numpy as np
import os
import glob

def parse_pdb_coordinates(pdb_file):
    """Extract coordinates and atom lines from PDB file"""
    coords = []
    atom_lines = []
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
                atom_lines.append(line)
                
    return np.array(coords), atom_lines

def write_centered_pdb(output_file, atom_lines, new_coords):
    """Write new PDB file with centered coordinates"""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("HEADER    CENTERED SMOTIF COORDINATES\n")
        f.write("REMARK    Coordinates centered to center of mass\n")
        
        for line, coord in zip(atom_lines, new_coords):
            new_line = (f"{line[:30]}"
                       f"{coord[0]:8.3f}"
                       f"{coord[1]:8.3f}"
                       f"{coord[2]:8.3f}"
                       f"{line[54:]}")
            f.write(new_line)
        f.write("END\n")

def center_coordinates(base_input_dir, base_output_dir):
    """Process all PDB files maintaining directory structure"""
    # Get all smotif type directories (HE_37_8, HH_11_29, etc.)
    smotif_dirs = glob.glob(os.path.join(base_input_dir, "*_*_*"))
    print(f"Found {len(smotif_dirs)} Smotif type directories")
    
    total_processed = 0
    for smotif_dir in smotif_dirs:
        # Get the smotif type folder name (e.g., "HE_37_8")
        smotif_type = os.path.basename(smotif_dir)
        
        # Create corresponding output directory
        output_smotif_dir = os.path.join(base_output_dir, smotif_type)
        os.makedirs(output_smotif_dir, exist_ok=True)
        
        # Find all PDB files in this smotif directory
        pdb_files = glob.glob(os.path.join(smotif_dir, "**/*.pdb"), recursive=True)
        print(f"Processing {smotif_type}: Found {len(pdb_files)} PDB files")
        
        processed = 0
        for pdb_file in pdb_files:
            try:
                # Get relative path within the smotif directory
                rel_path = os.path.relpath(pdb_file, smotif_dir)
                output_file = os.path.join(output_smotif_dir, rel_path)
                
                # Read coordinates
                coords, atom_lines = parse_pdb_coordinates(pdb_file)
                
                if len(coords) == 0:
                    print(f"Warning: No coordinates found in {pdb_file}")
                    continue
                
                # Calculate center of mass
                center_of_mass = coords.mean(axis=0)
                
                # Center coordinates
                centered_coords = coords - center_of_mass
                
                # Write new PDB file
                write_centered_pdb(output_file, atom_lines, centered_coords)
                
                processed += 1
                total_processed += 1
                if processed % 1000 == 0:
                    print(f"Processed {processed} files in {smotif_type}...")
                
            except Exception as e:
                print(f"Error processing {pdb_file}: {str(e)}")
                continue
        
        print(f"Completed {smotif_type}: Processed {processed} files")
    
    print(f"Successfully processed {total_processed} files total")

if __name__ == "__main__":
    # Example paths - replace with your actual paths
    input_directory = "/path/to/smotif/coordinates"  # Base directory containing HE_37_8, HH_11_29, etc.
    output_directory = "/path/to/centered/coordinates"  # Base directory for output
    
    center_coordinates(input_directory, output_directory)