import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import os
from numba_rmsd import calc_smotif_rmsd

def process_smotif_pair(args):
    # add deprcation warning
    smotif1, smotif2, pdb_dir = args
    pdb1 = os.path.join(pdb_dir, f"{smotif1['domain'][:7]}.pdb")
    pdb2 = os.path.join(pdb_dir, f"{smotif2['domain'][:7]}.pdb")
    
    
    coor_range = (smotif1['SSE1_start'], smotif1['SSE1_end'], smotif1['SSE2_start'], smotif1['SSE2_end'], smotif2['SSE1_start'], smotif2['SSE1_end'], smotif2['SSE2_start'], smotif2['SSE2_end'])
            
    rmsd = calc_smotif_rmsd(pdb1, pdb2, coor_range)
    print (f"RMSD between {smotif1['motif_id']} and {smotif2['motif_id']}: {rmsd}")
    return smotif1['motif_id'], smotif2['motif_id'], rmsd


def bin_smotifs(rmsd_df, df, rmsd_threshold=2.0, contact_threshold=0.2):
    bins = {}
    bin_id = 0
    
    for _, row in rmsd_df.iterrows():
        smotif1, smotif2 = row['pdb1'], row['pdb2']
        rmsd = row['rmsd']
        
        if rmsd <= rmsd_threshold:
            smotif1_data = df[df['motif_id'] == smotif1].iloc[0]
            smotif2_data = df[df['motif_id'] == smotif2].iloc[0]
            
            contact_diff = abs(smotif1_data['total_contacts'] - smotif2_data['total_contacts']) / max(smotif1_data['total_contacts'], smotif2_data['total_contacts'], 1)
            
            if contact_diff <= contact_threshold:
                # Check if either smotif is already in a bin
                bin1 = next((b for b, smotifs in bins.items() if smotif1 in smotifs), None)
                bin2 = next((b for b, smotifs in bins.items() if smotif2 in smotifs), None)
                
                if bin1 is None and bin2 is None:
                    # Create a new bin
                    bins[bin_id] = {smotif1, smotif2}
                    bin_id += 1
                elif bin1 is not None and bin2 is None:
                    # Add smotif2 to bin1
                    bins[bin1].add(smotif2)
                elif bin1 is None and bin2 is not None:
                    # Add smotif1 to bin2
                    bins[bin2].add(smotif1)
                elif bin1 != bin2:
                    # Merge the two bins
                    bins[bin1].update(bins[bin2])
                    del bins[bin2]
    
    return bins




def main():
    # Load the sample input CSV file
    input_file = 'test_extended_smotif_db_with_contacts.csv'
    df = pd.read_csv(input_file)
    #    create a new column called smotif_length which is the sum of sse1_length and sse2_length
    df['smotif_length'] = df['sse1_length'] + df['sse2_length']
    # split the test_extended_smotif_db_with_contacts.csv file into different files based on the smotif_type column
    smotif_types = df['smotif_type'].unique()
    smotif_dfs =[]
    # check if the files already exist if so, skip the process of creating the files
    for smotif_type in smotif_types:
        smotif_dfs.append(df[df['smotif_type'] == smotif_type])
        if os.path.exists(f'test_extended_smotif_db_with_contacts_{smotif_type}.csv'):
            continue
        else:
            df = df.sort_values(by=['smotif_length'], ascending=False)
            df[df['smotif_type'] == smotif_type].to_csv(f'test_extended_smotif_db_with_contacts_{smotif_type}.csv', index=False)    
    
    
    # Directory containing PDB files
    pdb_dir = '/home/kalabharath/projects/dingo_fold/cath_db/non-redundant-data-sets/dompdb/'  # Replace with actual path
    for df in smotif_dfs:
        
        # sort the df based on the length of SSE1 + SSE2
        df = df.sort_values(by=['smotif_length'], ascending=False)
        
        
        
        # Prepare arguments for multiprocessing
        smotif_pairs = [(row1, row2, pdb_dir) 
                        for i, row1 in df.iterrows() 
                        for row2 in df.iloc[i+1:].to_dict('records')]
        
        
        # Compute RMSD values using multiprocessing
        # c_count = cpu_count()
        # c_count = 1
        # with Pool(c_count) as pool:
        """
            results = list(tqdm(pool.imap(process_smotif_pair, smotif_pairs), 
                                total=len(smotif_pairs), 
                                desc="Computing RMSD"))
        """
        results = []    
        
        for smotif1, smotif2, pdb_dir in smotif_pairs:
            pdb1 = os.path.join(pdb_dir, f"{smotif1['domain'][:7]}.pdb")
            pdb2 = os.path.join(pdb_dir, f"{smotif2['domain'][:7]}.pdb")
            
            
            # implement sliding window here
            smotif_1 = [smotif1['SSE1_start'], smotif1['SSE1_end'], smotif1['SSE2_start'], smotif1['SSE2_end']]
            smotif_2 = [smotif2['SSE1_start'], smotif2['SSE1_end'], smotif2['SSE2_start'], smotif2['SSE2_end']] 
            print (f"Calculating RMSD between {smotif1['motif_id']} and {smotif2['motif_id']}")
            print (f"Coor Range: {coor_range}")
            window_pairs = sliding_window(coor_range, window_size=3, max_window_size=5)
            print (f"Number of window pairs: {len(window_pairs)}")
            print (window_pairs)
            
            rmsd_range = []
            
            for window_pair in window_pairs:
                trmsd = calc_smotif_rmsd(pdb1, pdb2, window_pair)
                rmsd_range.append(trmsd)
            
            rmsd = min(rmsd_range)
            print(f"RMSD between {smotif1['motif_id']} and {smotif2['motif_id']}: {rmsd}")
            results.append((smotif1['motif_id'], smotif2['motif_id'], rmsd))        
            
            sorted_rmsd_range = sorted(rmsd_range)
            rmsd = sorted_rmsd_range[0]
            print (f"RMSD between {smotif1['motif_id']} and {smotif2['motif_id']}: {rmsd}")
            results.append((smotif1['motif_id'], smotif2['motif_id'], rmsd))
        
        
        # Create DataFrame from results
        rmsd_df = pd.DataFrame(results, columns=['pdb1', 'pdb2', 'rmsd'])
        
        # Bin Smotifs based on RMSD and total contacts
        bins = bin_smotifs(rmsd_df, df)
        
        # Save binning results
        with open('smotif_binning_results.csv', 'w') as f:
            f.write("Bin_ID,Smotif_IDs,Smotif_Type,Total_Contacts\n")
            for bin_id, smotifs in bins.items():
                smotif_ids = ','.join(smotifs)
                smotif_type = df[df['motif_id'].isin(smotifs)]['smotif_type'].iloc[0]
                total_contacts = ','.join(map(str, df[df['motif_id'].isin(smotifs)]['total_contacts']))
                f.write(f"{bin_id},{smotif_ids},{smotif_type},{total_contacts}\n")
        
        print(f"Number of bins: {len(bins)}")
        
        # Plot distance map
        smotif_ids = df['motif_id'].tolist()
        
        n = len(smotif_ids)
        distance_map = np.zeros((n, n))
        
        for _, row in rmsd_df.iterrows():
            i = smotif_ids.index(row['pdb1'])
            j = smotif_ids.index(row['pdb2'])
            distance_map[i, j] = distance_map[j, i] = row['rmsd']
        
        plt.figure(figsize=(12, 10))
        plt.imshow(distance_map, cmap='viridis')
        plt.colorbar(label='RMSD')
        plt.title('RMSD Distance Map')
        plt.xlabel('Smotifs')
        plt.ylabel('Smotifs')
        plt.savefig('distance_map.png')
        plt.close()

if __name__ == "__main__":
    main()