import os, argparse, sys
import time
from src import post_process
import pickle
import pandas as pd

if __name__=="__main__":
    t=time.time()
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    main_subparsers = parser.add_subparsers(title="Options", dest="option")
    
    parent_parser = argparse.ArgumentParser(add_help=False,)
    parent_parser.add_argument("--gff", help='GFF file', type=str, required=True)
    parent_parser.add_argument("--output", help='Output folder', type=str)
    parent_parser.add_argument("--prefix", help='Output file prefix', type=str, default="output")
    parent_parser.add_argument("--unannotated", help='BED file of unannotated regions', type=str)
    parent_parser.add_argument("--distance_threshold", help='Distance threshold for merging breakpoints', type=int, default=10) 
    parent_parser.add_argument("--min_support", help='Minimum read support for reporting gene fusion', type=int, default=2)
    
    
    detect_parser = main_subparsers.add_parser("detect", parents=[parent_parser],
                                      add_help=True,
                                      help="Detect gene fusions and novel isoforms from BAM file.",  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    detect_required=detect_parser.add_argument_group("Required Arguments")
    detect_parser.add_argument("--bam", help='Path to aligned BAM file sorted by name.', type=str, required=True)
    detect_parser.add_argument("--check_strand", help='Check strand orientation of reads using ply A tail and primers', default=False, action='store_true')
    detect_parser.add_argument("--threads", help='Number of threads', type=int, default=4)
    detect_parser.add_argument("--gf_only", help='Check gene fusions only', default=False, action='store_true')
     
    
    merge_filter_parser = main_subparsers.add_parser("merge_filter", parents=[parent_parser],
                                      add_help=True,
                                      help="Merge and filter gene fusions using custom parameters using pre-computed read level pickled files.",  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    merge_filter_parser.add_argument('--pickle', help="Read level pickled files from detect module", required=True)
    
    args = parser.parse_args()
    
    if not args.output:
        args.output=os.getcwd()
    
    output_path=args.output
    os.makedirs(output_path, exist_ok=True)
    
    with open(os.path.join(args.output,'args'),'w') as file:
            file.write('Command: python %s\n\n\n' %(' '.join(sys.argv)))
            file.write('------Parameters Used------\n')
            for k in vars(args):
                file.write('{}: {}\n'.format(k,vars(args)[k]))
                
    if args.option=='detect':
        from src import detect
        total_output, output, gene_id_to_name=detect.call_manager(args)
        
    else:
        with open(args.pickle, 'rb') as handle:
            output=pickle.load(handle)
        
        gene_df=pd.read_csv(args.gff, sep='\t', comment='#', header=None)
        gene_df.rename(columns={0:'chrom', 2:'feature', 3:'start', 4:'end',6:'strand', 8:'info'}, inplace=True)
        gene_df['gene_id']=gene_df['info'].str.extract(r'gene_id=([^;]+)')
        gene_df['gene_name']=gene_df['info'].str.extract(r'gene_name=([^;]+)')
        gene_df=gene_df[gene_df.feature=='gene']
        gene_id_to_name={x:y for x,y in zip(gene_df.gene_id, gene_df.gene_name)}
        
        
    final_gf_double_bp, final_gf_single_bp=post_process.get_GFs(output, gene_id_to_name, args.distance_threshold, args.min_support)
    
    with open(os.path.join(output_path,args.prefix+'.final_gf_double_bp.pickle'), 'wb') as handle:
        pickle.dump(final_gf_double_bp, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(output_path,args.prefix+'.final_gf_single_bp.pickle'), 'wb') as handle:
        pickle.dump(final_gf_single_bp, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(output_path,'.final_gf_double_bp'), 'w') as ann_file, open(os.path.join(output_path,'.final_gf_double_bp.inconsistent'), 'w') as incon_file:
        for k,v in sorted(final_gf_double_bp.items(), key=lambda x: x[1]['read_support'], reverse=True):
            if v['consistent']:
                ann_file.write("{}::{} Support={} {}:{}({}bp, {}q, {}bp, {}) {}:{}({}bp, {}q, {}bp, {}) ('{}','{}') Annotated={}  Readthrough={}   Genes Overlap={}\n".format(post_process.get_gene_name(k[0], gene_id_to_name), post_process.get_gene_name(k[1], gene_id_to_name), v['read_support'], *v['median_breakpoint_1'][:2], *v['median_breakpoint_1'][3:], *v['median_breakpoint_2'][:2], *v['median_breakpoint_2'][3:], v['median_breakpoint_1'][2], v['median_breakpoint_2'][2], v['annotated'],v['readthrough'], v['genes_overlap']))

                
            else:
                incon_file.write("{}::{} Support={} {}:{}({}bp, {}q, {}bp, {}) {}:{}({}bp, {}q, {}bp, {}) ('{}','{}') Annotated={}  Readthrough={}   Genes Overlap={}\n".format(post_process.get_gene_name(k[0], gene_id_to_name), post_process.get_gene_name(k[1], gene_id_to_name), v['read_support'], *v['median_breakpoint_1'][:2], *v['median_breakpoint_1'][3:], *v['median_breakpoint_2'][:2], *v['median_breakpoint_2'][3:], v['median_breakpoint_1'][2], v['median_breakpoint_2'][2], v['annotated'],v['readthrough'], v['genes_overlap']))
                
    print('Time elapsed={}s'.format(time.time()-t))