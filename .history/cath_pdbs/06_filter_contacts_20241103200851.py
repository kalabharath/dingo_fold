import pandas as pd
import os




def main(csv_file):
    df = pd.read_csv(csv_file)
    # print total number of entries
    print(f"Total number of entries: {len(df)}")
    total_entries = len(df) 
    # delete all rows where where sse1_length or sse2_length is less than 4
    
    df = df[df['sse1_length'] >= 4]
    # print lost entries
    print(f"Lost entries after sse1 length: {total_entries - len(df)}")
    total_entries = len(df) 
    df = df[df['sse2_length'] >= 4]
    print (f"Lost entries after sse2 length: {total_entries - len(df)}")
    # delete all rows where loop_length is negative
    total_entries = len(df) 
    df = df[df['loop_length'] >= 1]
    print (f"Lost entries after loop length: {total_entries - len(df)}")
    # delete all rows where total_contacts is zero 'and' loop_length is is greater than 10 
    total_entries = len(df)
    df = df.drop(df[(df['total_contacts'] < 5) & (df['loop_length'] > 12)].index)    
    print (f"Lost entries after total_contacts and loop_length: {total_entries - len(df)}")
    # save the filtered dataframe
    output_file = "filtered_" + csv_file
    df.to_csv(output_file, index=False)
    print(f"Filtered data saved to {output_file}")  
    
    
if __name__ == "__main__":
    main("extended_smotif_db_with_contacts.csv")