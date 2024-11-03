import requests
import os
import time
from pathlib import Path
import pandas as pd

def download_pdb(domain_id, output_dir, base_url="http://www.cathdb.info/version/v4_3_0/api/rest/id"):
    """
    Download a PDB file for a given domain ID from CATH database

    Args:
        domain_id (str): The CATH domain ID
        output_dir (str): Directory to save the PDB file
        base_url (str): Base URL for the CATH REST API

    Returns:
        bool: True if download successful, False otherwise
    """
    # Clean the domain ID (remove any whitespace)
    domain_id = domain_id.strip()

    # Construct the full URL
    url = f"{base_url}/{domain_id}.pdb"

    # Create output filename
    output_file = os.path.join(output_dir, f"{domain_id}.pdb")

    # Skip if file already exists
    if os.path.exists(output_file):
        print(f"Skipping {domain_id}, file already exists")
        return True

    try:
        # Make the request
        response = requests.get(url)

        # Check if request was successful
        if response.status_code == 200:
            # Save the PDB file
            with open(output_file, 'w') as f:
                f.write(response.text)
            print(f"Successfully downloaded {domain_id}")
            return True
        else:
            print(f"Failed to download {domain_id}. Status code: {response.status_code}")
            return False

    except Exception as e:
        print(f"Error downloading {domain_id}: {str(e)}")
        return False

def main():
    # Create output directory if it doesn't exist
    output_dir = "pdb_files"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Read the input file
    # Using whitespace as delimiter and no header
    try:
        df = pd.read_csv('/home/kalabharath/projects/dingo_fold/cath_db/cath-classification-data/cath-domain-list-S100.txt', delim_whitespace=True, header=None)
        domain_ids = df[0].tolist()  # First column contains domain IDs
    except Exception as e:
        print(f"Error reading input file: {str(e)}")
        return

    print(f"Found {len(domain_ids)} domain IDs to process")

    # Download each PDB file
    successful = 0
    failed = 0

    for i, domain_id in enumerate(domain_ids, 1):
        print(f"\nProcessing {i}/{len(domain_ids)}: {domain_id}")

        if download_pdb(domain_id, output_dir):
            successful += 1
        else:
            failed += 1

        # Add a small delay to avoid overwhelming the server
        time.sleep(1)

    # Print summary
    print("\nDownload Summary:")
    print(f"Total attempted: {len(domain_ids)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

if __name__ == "__main__":
    main()
