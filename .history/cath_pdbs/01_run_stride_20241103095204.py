import sys, os, glob


def run_stride(domain, output_file):
    run_stride = "stride -f "+domain+" >"+output_file+".stride"
    return os.system(run_stride)


if __name__ == '__main__':
    domains = glob.glob("/home/kalabharath/projects/dingo_fold/cath_db/non-redundant-data-sets/dompdb/*")
    output_dir = '/home/kalabharath/projects/dingo_fold/cath_db/smotif_db/stride_annotations'
    
    for domain in domains:
        domain_name = domain.split('/')[-1]
        output_file = os.path.join(output_dir, domain_name)
        run_stride(domain, output_file)
        print(f"Processed {domain_name}")
                        
    