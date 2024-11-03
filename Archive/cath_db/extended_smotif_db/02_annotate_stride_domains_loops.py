import os
from collections import namedtuple
import csv
import logging
from datetime import datetime

# Set up logging
log_filename = f"smotif_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_filename),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

SSE = namedtuple('SSE', ['type', 'start', 'end', 'length'])
SSEPair = namedtuple('SSEPair', ['sse1', 'sse2', 'loop_length'])

def process_stride_file(stride_file):
    """
    Process a STRIDE file to identify pairs of secondary structure elements,
    verifying continuity and applying minimum length requirements based on the file content.
    
    :param stride_file: Path to the STRIDE file
    :return: List of SSEPair namedtuples
    """
    def is_helix(ss_type):
        return ss_type in ['H', 'G', 'I']  # AlphaHelix, 310Helix, PiHelix

    def simplify_ss_type(ss_type):
        if is_helix(ss_type):
            return 'H'  # Helix
        elif ss_type == 'E':
            return 'E'  # Extended (beta strand)
        else:
            return 'C'  # Coil (including turns, bridges, and undefined structures)

    def meets_length_requirement(sse):
        return (sse.type == 'H' and sse.length >= 4) or (sse.type == 'E' and sse.length >= 3)

    def is_continuous(start, end, residue_numbers):
        return all(res in residue_numbers for res in range(start, end + 1))

    sses = []
    current_sse = None
    residue_numbers = set()
    residue_ss_map = {}
    
    logger.info(f"Processing file: {stride_file}")
    
    with open(stride_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.startswith('ASG'):
                parts = line.split()
                if len(parts) < 6:
                    logger.warning(f"Line {line_num} is not in the expected format: {line.strip()}")
                    continue
                
                residue_id = parts[3]
                ss_type = parts[5]
                simple_type = simplify_ss_type(ss_type)
                
                # Ignore residues with insertion codes
                if not residue_id.isdigit():
                    logger.debug(f"Ignoring residue with insertion code: {residue_id}")
                    continue

                residue_num = int(residue_id)
                residue_numbers.add(residue_num)
                residue_ss_map[residue_num] = simple_type

                if simple_type in ['H', 'E']:
                    if current_sse and current_sse.type == simple_type and residue_num == current_sse.end + 1:
                        # Extend current SSE
                        current_sse = SSE(simple_type, current_sse.start, residue_num, residue_num - current_sse.start + 1)
                    else:
                        # Start new SSE
                        if current_sse and meets_length_requirement(current_sse) and is_continuous(current_sse.start, current_sse.end, residue_numbers):
                            sses.append(current_sse)
                        current_sse = SSE(simple_type, residue_num, residue_num, 1)
                else:
                    # Coil region, end current SSE if exists
                    if current_sse and meets_length_requirement(current_sse) and is_continuous(current_sse.start, current_sse.end, residue_numbers):
                        sses.append(current_sse)
                    current_sse = None

    # Add last SSE if exists, meets length requirement, and is continuous
    if current_sse and meets_length_requirement(current_sse) and is_continuous(current_sse.start, current_sse.end, residue_numbers):
        sses.append(current_sse)

    logger.info(f"Total SSEs found: {len(sses)}")

    # Create pairs of consecutive SSEs
    sse_pairs = []
    for i in range(len(sses) - 1):
        sse1 = sses[i]
        sse2 = sses[i + 1]
        # Verify loop continuity and positive loop length
        loop_length = sse2.start - sse1.end - 1
        if loop_length < 0:
            logger.warning(f"Negative loop length detected between SSEs: {sse1} and {sse2}")
            continue
        
        loop_continuous = is_continuous(sse1.end + 1, sse2.start - 1, residue_numbers)
        loop_is_coil = all(residue_ss_map.get(res, 'C') == 'C' for res in range(sse1.end + 1, sse2.start))
        
        # Only create pairs if both elements are helices or strands,
        # meet the minimum length requirements,
        # the loop is continuous, and consists only of coil residues
        if (sse1.type in ['H', 'E'] and sse2.type in ['H', 'E'] and 
            meets_length_requirement(sse1) and meets_length_requirement(sse2) and
            loop_continuous and loop_is_coil):
            sse_pairs.append(SSEPair(sse1, sse2, loop_length))
            logger.debug(f"Added SSE pair: {sse1} - {sse2}, Loop length: {loop_length}")

    logger.info(f"Total SSE pairs found: {len(sse_pairs)}")
    return sse_pairs

def process_domains(stride_dir, output_file):
    """
    Process all STRIDE files in the given directory and save Smotif information to a CSV file.
    
    :param stride_dir: Directory containing STRIDE files
    :param output_file: Path to the output CSV file
    """
    logger.info(f"Starting to process domains in directory: {stride_dir}")
    logger.info(f"Output will be written to: {output_file}")

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Domain', 'SSE1_Type', 'SSE1_Start', 'SSE1_End', 'SSE1_Length',
                         'SSE2_Type', 'SSE2_Start', 'SSE2_End', 'SSE2_Length', 'Loop_Length'])
        
        total_pairs = 0
        processed_files = 0
        for filename in os.listdir(stride_dir):
            if filename.endswith('.stride'):
                processed_files += 1
                domain = filename.split('_')[0]
                stride_file = os.path.join(stride_dir, filename)
                sse_pairs = process_stride_file(stride_file)
                
                for pair in sse_pairs:
                    writer.writerow([
                        domain,
                        pair.sse1.type, pair.sse1.start, pair.sse1.end, pair.sse1.length,
                        pair.sse2.type, pair.sse2.start, pair.sse2.end, pair.sse2.length,
                        pair.loop_length
                    ])
                    total_pairs += 1

                if processed_files % 100 == 0:
                    logger.info(f"Processed {processed_files} files. Current total Smotifs: {total_pairs}")

        logger.info(f"Processing complete. Total files processed: {processed_files}")
        logger.info(f"Total Smotifs written to CSV: {total_pairs}")

# Example usage
stride_dir = './stride_annotations/'
output_file = 'smotif_database2.csv'

logger.info("Script execution started")
process_domains(stride_dir, output_file)
logger.info("Script execution completed")