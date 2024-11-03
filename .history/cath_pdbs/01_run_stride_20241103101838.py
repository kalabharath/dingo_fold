import sys
import os
import glob
from multiprocessing import Pool
from functools import partial
import tqdm

def run_stride(output_dir, domain):
    """
    Run STRIDE on a single domain
    
    Args:
        output_dir (str): Directory to save output files
        domain (str): Path to input PDB file
    """
    domain_name = domain.split('/')[-1]
    output_file = os.path.join(output_dir, domain_name)
    run_stride_cmd = f"stride -f {domain} >{output_file}.stride"
    return_code = os.system(run_stride_cmd)
    return domain_name, return_code

def process_domains_parallel(domains, output_dir, n_processors=None):
    """
    Process multiple domains in parallel using a multiprocessing pool
    
    Args:
        domains (list): List of domain paths to process
        output_dir (str): Directory to save output files
        n_processors (int, optional): Number of processors to use. Defaults to CPU count.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # If n_processors not specified, use all available CPUs
    if n_processors is None:
        n_processors = os.cpu_count()
    
    # Create partial function with fixed output_dir
    process_func = partial(run_stride, output_dir)
    
    # Create process pool and run processing
    with Pool(processes=n_processors) as pool:
        # Use tqdm to show progress bar
        results = list(tqdm.tqdm(
            pool.imap(process_func, domains),
            total=len(domains),
            desc="Processing domains"
        ))
    
    # Check for any failures
    failures = [(domain, code) for domain, code in results if code != 0]
    if failures:
        print("\nThe following domains failed processing:")
        for domain, code in failures:
            print(f"- {domain} (return code: {code})")
    
    return results

if __name__ == '__main__':
    # Define input and output paths
    domains = glob.glob('/home/kalabharath/projects/dingo_fold/cath_pdbs/pdb_files/*.pdb')
    output_dir = 'stride_annotations'
    
    # Process all domains in parallel
    results = process_domains_parallel(
        domains=domains,
        output_dir=output_dir,
        n_processors=None  # Uses all available CPUs by default
    )
    
    # Print summary
    total = len(domains)
    successful = sum(1 for _, code in results if code == 0)
    print(f"\nProcessing complete!")
    print(f"Successfully processed {successful}/{total} domains")