import numpy.lib.recfunctions as rfn
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import numpy_indexed as npi
import re, pysam, random
import multiprocessing as mp
import parasail
from .utils import *

sub_mat=parasail.matrix_create('AGTC',10,-10)

def get_longest_match(seq, ref):
    global sub_mat
    alignment=parasail.sw_trace_striped_sat(s1=seq, s2=ref, open=40, extend=10, matrix=sub_mat)
    matches=[x >> 4 for x in alignment.cigar.seq if x & 0xf ==7]
    max_len=max(matches)
    sum_match=sum(matches)
    return alignment.score, max_len, sum_match

def get_pval(seq, ref,n=100):
    global sub_mat
    rev_seq=revcomp(seq)
    
    score, max_len, sum_match=get_longest_match(seq, ref)
    score_rev, max_len_rev, sum_match_rev=get_longest_match(rev_seq, ref)
    
    best_score, seq, best_max_len, best_sum_match= (score, seq, max_len, sum_match) if score >score_rev else (score_rev, rev_seq, max_len_rev, sum_match_rev)

    tests=np.array([parasail.sw_striped_sat(s1=''.join(random.sample(seq, len(seq))), s2=ref, open=40, extend=10, matrix=sub_mat).score for i in range(n)])

    return  best_sum_match, best_max_len, best_score, np.mean(tests>=best_score), (best_score-np.mean(tests))/np.std(tests)
        
def split_list(l,n=100):
    i=0    
    chunk = l[i*n:(i+1)*n]
    while chunk:
        yield chunk
        i+=1
        chunk = l[i*n:(i+1)*n]

def get_strand(gene_id, gene_strand_map):
    try:
        return gene_strand_map[gene_id]
    except KeyError:
        return '+-'
    
def get_gene_name(gene_id, gene_id_to_name):
    try:
        return gene_id_to_name[gene_id]
    except KeyError:
        return gene_id
    
def match_strand(gene_strand_1, gene_strand_2, read_strand_1, read_strand_2, switch=False):
    strand_switch={'+':'-', '-':'+'}
    if switch:
        read_strand_1, read_strand_2=strand_switch[read_strand_1], strand_switch[read_strand_2]
        
    gene_1_pass=True if gene_strand_1=='+-' else gene_strand_1==read_strand_1
    gene_2_pass=True if gene_strand_2=='+-' else gene_strand_2==read_strand_2
    return gene_1_pass and gene_2_pass

def parse_GF_output(data, gene_id_to_name, gene_strand_map, inv_chrom_map, inv_strand_map, overlapping_genes):
    output={}
    genes_overlap_output={}
    total_output={}    

    for x in data:
        rname, init_gene_list, rbp, bp, strands, mapq, bp_intron, same_segment, read_seq=x

        gene_list=tuple(sorted(init_gene_list ))#(gene_map[v] for v in gene_list)
        genes_overlap=gene_list in overlapping_genes

        total_output[rname]=dict(gene_list=init_gene_list, read_breakpoints=rbp, reference_breakpoints=bp, read_strands=strands, genes_overlap=genes_overlap, mapq=mapq, bp_intron=bp_intron)

        gene_1, gene_2 = init_gene_list[0], init_gene_list[1]
        chrom_1=inv_chrom_map[bp[0][0]]
        chrom_2=inv_chrom_map[bp[1][0]]
        bp1, bp2 = bp[0][1], bp[1][1]
        mapq1, mapq2=mapq
        bp1_intron, bp2_intron=bp_intron

        rbp1, rbp2=list(rbp[0]), list(rbp[1])

        read_strand_1, read_strand_2=inv_strand_map[strands[0]], inv_strand_map[strands[1]]
        gene_strand_1, gene_strand_2=get_strand(gene_1, gene_strand_map), get_strand(gene_2, gene_strand_map) 
        
        gene_1_data=(rbp1[0],rbp1[1], read_strand_1, chrom_1, bp1, gene_1, get_gene_name(gene_1, gene_id_to_name), gene_strand_1, mapq1, bp1_intron)
        gene_2_data=(rbp2[0],rbp2[1], read_strand_2, chrom_2, bp2, gene_2, get_gene_name(gene_2, gene_id_to_name), gene_strand_2, mapq2, bp2_intron)

        if match_strand(gene_strand_1, gene_strand_2, read_strand_1, read_strand_2, switch=False):
            gene_list=(gene_1, gene_2, 'consistent')

        elif match_strand(gene_strand_1, gene_strand_2, read_strand_1, read_strand_2, switch=True):
            gene_1_data, gene_2_data= gene_2_data, gene_1_data
            gene_list=(gene_2, gene_1, 'consistent')

        else:
            gene_list=(gene_1, gene_2,'inconsistent')

        if gene_list not in output:
            output[gene_list]={'reads':[], 'info':[],'read_seqs':[],'genes_overlap':genes_overlap, 'is_consistent': gene_list[2]=='consistent', 'same_segment':same_segment}
        output[gene_list]['reads'].append(rname)
        output[gene_list]['info'].append((gene_1_data, gene_2_data))
        output[gene_list]['read_seqs'].append(read_seq)
        
    
    return total_output, output

def get_cluster(args):
    read1, read2, read_names, genes_overlap, consistent, same_segment, gene_id_to_name, gene1_ref_seq, gene2_ref_seq, read_seqs, distance_threshold, min_support=args
    
    data_to_fit=np.vstack([read1['bp'], read2['bp']]).T


    cluster_dict={}
    if len(read_names)>1:       
        cluster_alg = AgglomerativeClustering(n_clusters =None, distance_threshold=distance_threshold, metric='manhattan', linkage='single').fit(data_to_fit)
        cluster_labels=cluster_alg.labels_
    else:
        cluster_labels=np.array([0])

    cluster_group=npi.group_by(cluster_labels)
    split_cluster=cluster_group.split(np.arange(len(cluster_labels)))

    cluster_dict={}

    i=1
    for cluster_indices in sorted(split_cluster, key=lambda x: len(x), reverse=True):
        read_support=len(cluster_indices)
        if read_support>=min_support:            
            read1_cluster, read2_cluster, cluster_read_names =read1[cluster_indices], read2[cluster_indices], read_names[cluster_indices]

            mapq1, mapq2=np.median(read1_cluster['mapq']).astype(int), np.median(read2_cluster['mapq']).astype(int).astype(int)
            
            bp1, bp2=np.median(read1_cluster['bp']).astype(int), np.median(read2_cluster['bp']).astype(int)
            
            bp1_range, bp2_range=np.max(read1_cluster['bp'])-np.min(read1_cluster['bp']), np.max(read2_cluster['bp'])-np.min(read2_cluster['bp'])

            r1len =np.max(read1_cluster['read_bp_3p']-read1_cluster['read_bp_5p'])+1
            r2len =np.max(read2_cluster['read_bp_3p']-read2_cluster['read_bp_5p'])+1
            
            intron1="Intronic" if np.mean(read1_cluster['intron'])>=0.5 else "Exonic"
            intron2="Intronic" if np.mean(read2_cluster['intron'])>=0.5 else "Exonic"
            
            chrom_1=read1_cluster['chrom'][0]
            chrom_2=read2_cluster['chrom'][0]
            gene_1=read1_cluster['gene_id'][0]
            gene_2=read2_cluster['gene_id'][0]
            
            best_read1_idx=np.argmax(read1_cluster['read_bp_3p']-read1_cluster['read_bp_5p'])
            best_read2_idx=np.argmax(read2_cluster['read_bp_3p']-read2_cluster['read_bp_5p'])
                        
            read1_seq=read_seqs[cluster_indices][best_read1_idx][read1_cluster['read_bp_5p'][best_read1_idx]-1:read1_cluster['read_bp_3p'][best_read1_idx]]
            read2_seq=read_seqs[cluster_indices][best_read2_idx][read2_cluster['read_bp_5p'][best_read2_idx]-1:read2_cluster['read_bp_3p'][best_read2_idx]]
            
            #best_sum_match, best_max_len, best_score, np.mean(tests>=best_score), (best_score-np.mean(tests))/np.std(tests)
                       
            gene1_sig=get_pval(read2_seq, gene1_ref_seq,n=100) if len(gene1_ref_seq)<=1e6 else [-1,-1,-1,-1,-1]
            gene1_sig={'num_matches':gene1_sig[0], 'max_match':gene1_sig[1], 'score':gene1_sig[2], 'pval':gene1_sig[3], 'zscore':gene1_sig[4]}
            gene2_sig=get_pval(read1_seq,gene2_ref_seq,n=100)  if len(gene2_ref_seq)<=1e6 else [-1,-1,-1,-1,-1]
            gene2_sig={'num_matches':gene2_sig[0], 'max_match':gene2_sig[1], 'score':gene2_sig[2], 'pval':gene2_sig[3], 'zscore':gene2_sig[4]}
            
            readthrough=True if chrom_1==chrom_2 and abs(bp1-bp2)<200000 and same_segment else False
            
            annotated=(gene_1 in gene_id_to_name) + (gene_2 in gene_id_to_name)
            value={'median_breakpoint_1':(read1_cluster[0][6], gene_1, chrom_1, bp1, bp1_range, mapq1, r1len, intron1), \
                                              'median_breakpoint_2':(read2_cluster[0][6], gene_2, chrom_2, bp2 , bp2_range, mapq2, r2len, intron2), \
                                              'read_support': read_support,'annotated':annotated, \
                                              'read_names': cluster_read_names,\
                                               'read1_cluster':read1_cluster, 'read2_cluster':read2_cluster,\
                                              'genes_overlap':genes_overlap, 'consistent':consistent, 'readthrough': readthrough,
                                              'gene1_sig':gene1_sig, 'gene2_sig':gene2_sig}

            cluster_dict[(gene_1, gene_2,'GF{}_{}'.format('' if consistent else '_inconsistent',i))]=value
            i+=1
    return cluster_dict.copy()

    
def get_GFs(output, gene_id_to_name, ref_fasta, gene_df, distance_threshold=20, min_support=1, threads=1):
    gref=pysam.FastaFile(ref_fasta)
    
    
    final_gf_double_bp={}
    
    with mp.Pool(threads) as pool:
        input_list=[]

        for key in output.keys():
            #dtype=[('read_bp_5p', int), ('read_bp_3p', int), ('read_strand', 'O'), ('chrom', 'O'), ('bp', int), ('gene_id','O'), ('gene_name', 'O'), ('gene_strand', 'O'), ('mapq',int), ('intron', bool)])
            read1=np.array([x[0] for x in output[key]['info']], dtype=[('read_bp_5p', int), ('read_bp_3p', int), ('read_strand', 'O'), ('chrom', 'O'), ('bp', int), ('gene_id','O'), ('gene_name', 'O'), ('gene_strand', 'O'), ('mapq',int), ('intron', bool)])
            read2=np.array([x[1] for x in output[key]['info']], dtype=[('read_bp_5p', int), ('read_bp_3p', int), ('read_strand', 'O'), ('chrom', 'O'), ('bp', int), ('gene_id','O'), ('gene_name', 'O'), ('gene_strand', 'O'), ('mapq',int), ('intron', bool)])

            gc,gs,ge=gene_df[gene_df.gene_id==read1['gene_id'][0]][['chrom','start', 'end']].iloc[0]
            gene1_ref_seq=gref.fetch(gc,gs-1,ge)

            gc,gs,ge=gene_df[gene_df.gene_id==read2['gene_id'][0]][['chrom','start', 'end']].iloc[0]
            gene2_ref_seq=gref.fetch(gc,gs-1,ge)

            read_names=np.array(output[key]['reads'])
            read_seqs=np.array(output[key]['read_seqs'])
            genes_overlap, consistent, same_segment=output[key]['genes_overlap'], output[key]['is_consistent'], output[key]['same_segment']
            input_list.append([read1, read2, read_names, genes_overlap, consistent, same_segment, gene_id_to_name, gene1_ref_seq, gene2_ref_seq, read_seqs, distance_threshold, min_support])

        results=pool.imap_unordered(get_cluster, input_list)

        for double_bp in results:
            final_gf_double_bp.update(double_bp)
            
    return final_gf_double_bp
