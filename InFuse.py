import os, argparse, sys, datetime
import time
from src import post_process
import pickle
import pandas as pd

if __name__=="__main__":
    t=time.time()
    
    print('{}: Starting InFuse.'.format(str(datetime.datetime.now())))
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    main_subparsers = parser.add_subparsers(title="Options", dest="option")
    
    parent_parser = argparse.ArgumentParser(add_help=False,)
    parent_parser.add_argument("--gff", help='GFF file', type=str, required=True)
    parent_parser.add_argument("--output", help='Output folder', type=str)
    parent_parser.add_argument("--prefix", help='Output file prefix', type=str, default="output")
    parent_parser.add_argument("--unannotated", help='BED file of unannotated regions', type=str)
    parent_parser.add_argument("--distance_threshold", help='Distance threshold for merging breakpoints', type=int, default=10) 
    parent_parser.add_argument("--min_support", help='Minimum read support for reporting gene fusion', type=int, default=2)
    parent_parser.add_argument("--transcripts", help='Transcripts FASTA file', type=str, default='')
    
    
    detect_parser = main_subparsers.add_parser("detect", parents=[parent_parser],
                                      add_help=True,
                                      help="Detect gene fusions and novel isoforms from BAM file.",  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    detect_required=detect_parser.add_argument_group("Required Arguments")
    detect_parser.add_argument("--bam", help='Path to aligned BAM file.', type=str, required=True)
    detect_parser.add_argument("--check_strand", help='Check strand orientation of reads', default=False, action='store_true')
    detect_parser.add_argument("--threads", help='Number of threads', type=int, default=4)
    detect_parser.add_argument("--gf_only", help='Check gene fusions only', default=False, action='store_true')
     
    
    merge_filter_parser = main_subparsers.add_parser("merge_filter", parents=[parent_parser],
                                      add_help=True,
                                      help="Merge and filter gene fusions using custom parameters using pre-computed read level pickled files.",  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    merge_filter_parser.add_argument('--pickle', help="Read level pickled files from detect module", required=True)
        
    print('Command: python %s\n' %(' '.join(sys.argv)))
    
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
        total_output, output, gene_id_to_name, gene_df, gene_transcript_map, gene_strand_map, get_gene_exons, non_coding_gene_id, raw_exons = detect.call_manager(args)
        
        
    else:
        with open(args.pickle, 'rb') as handle:
            output=pickle.load(handle)
        
        gene_df=pd.read_csv(args.gff, sep='\t', comment='#', header=None)
        gene_df.rename(columns={0:'chrom', 2:'feature', 3:'start', 4:'end',6:'strand', 8:'info'}, inplace=True)
        gene_df['gene_id']=gene_df['info'].str.extract(r'gene_id=([^;]+)')[0].str.extract(r"([^.]+)")
        gene_df['transcript_id']=gene_df['info'].str.extract(r'transcript_id=([^;]+)')[0].str.extract(r"([^.]+)")
        gene_df['gene_name']=gene_df['info'].str.extract(r'gene_name=([^;]+)')

        gene_tr_df=gene_df[['gene_id', 'transcript_id']].dropna().groupby('gene_id')['transcript_id'].apply(lambda x: sorted(list(set(x)))).reset_index()
        
        gene_transcript_map={x:y for x,y in zip(gene_tr_df['gene_id'], gene_tr_df['transcript_id'])}
        
        gene_df=gene_df[gene_df.feature=='gene']
        gene_id_to_name={x:y for x,y in zip(gene_df.gene_id, gene_df.gene_name)}
        
    
    print('{}: Clustering gene fusions.'.format(str(datetime.datetime.now())))
    final_gf_double_bp=post_process.get_GFs(output, gene_transcript_map, args.transcripts, gene_df, non_coding_gene_id, args.distance_threshold, args.min_support, args.threads)
    
    print('{}: Saving gene fusions results.'.format(str(datetime.datetime.now())))
    
    with open(os.path.join(output_path,args.prefix+'.final_gf_double_bp.pickle'), 'wb') as handle:
        pickle.dump(final_gf_double_bp, handle, protocol=pickle.HIGHEST_PROTOCOL)

    
    header="\t".join(["gene_fusion", "read_support", "genes_overlap", "consistent", "readthrough",\
                  "gene_1_name", "gene_1_id", "chr_bp1", "pos_bp1", "range_bp1", "mapq_bp1", "max_len_bp1", "region_type_bp1", "gene1_best_transcript", "gene1_transcript_mapq",\
                  "gene_2_name", "gene_2_id", "chr_bp2", "pos_bp2", "range_bp2", "mapq_bp2", "max_len_bp2", "region_type_bp2", "gene2_best_transcript", "gene2_transcript_mapq",\
                  'gene1_sig_num_matches', 'gene1_sig_max_match', 'gene1_sig_score', 'gene1_sig_pval', 'gene1_sig_zscore',\
                  'gene2_sig_num_matches', 'gene2_sig_max_match', 'gene2_sig_score', 'gene2_sig_pval', 'gene2_sig_zscore'])
    
    with open(os.path.join(output_path,args.prefix+'.final_gf_double_bp'), 'w') as ann_file:
        ann_file.write(header+'\n')
        
        for k,v in sorted(final_gf_double_bp.items(), key=lambda x: x[1]['read_support'], reverse=True):
            gene_fusion="{}::{}".format(v['median_breakpoint_1'][0], v['median_breakpoint_2'][0])
            read_support, genes_overlap, consistent, readthrough= v['read_support'], v['genes_overlap'], v['consistent'], v['readthrough']
            rec="\t".join(str(x) for x in [gene_fusion, read_support, genes_overlap, consistent, readthrough, *v['median_breakpoint_1'], *v['gene1_best_transcript'], *v['median_breakpoint_2'], *v['gene2_best_transcript'], *v['gene1_sig'].values(), *v['gene2_sig'].values()])
            ann_file.write(rec+'\n')
            
            
    if not args.gf_only:
        final_skips, single_transcript_skips, single_transcript_non_repeat_non_skips, single_transcript_repeat_non_skips, multi_transcript_isoforms, reads_to_check=post_process.process_isoforms(args.bam, gene_strand_map, raw_exons, get_gene_exons, args.transcripts)
        
        print('{}: Saving novel isoforms results.'.format(str(datetime.datetime.now())))
        with open(os.path.join(output_path,args.prefix+'.single_transcript_skips.pickle'), 'wb') as handle:
            pickle.dump(single_transcript_skips, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(os.path.join(output_path,args.prefix+'.single_transcript_non_repeat_non_skips.pickle'), 'wb') as handle:
            pickle.dump(single_transcript_non_repeat_non_skips, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(os.path.join(output_path,args.prefix+'.single_transcript_repeat_non_skips.pickle'), 'wb') as handle:
            pickle.dump(single_transcript_repeat_non_skips, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(os.path.join(output_path,args.prefix+'.multi_transcript_isoforms.pickle'), 'wb') as handle:
            pickle.dump(multi_transcript_isoforms, handle, protocol=pickle.HIGHEST_PROTOCOL)     
        
        with open(os.path.join(output_path,args.prefix+'.final_skips.pickle'), 'wb') as handle:
            pickle.dump(final_skips, handle, protocol=pickle.HIGHEST_PROTOCOL)    

    print('Time elapsed={}s'.format(time.time()-t))
