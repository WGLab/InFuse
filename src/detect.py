import re, os, argparse

import multiprocessing as mp
import time

from ncls import NCLS64
import pickle
import queue
import numpy_indexed as npi
import parasail
from collections import Counter

import pandas as pd
import pysam
import numpy as np
from numba import jit
import itertools
import collections

from .post_process import parse_GF_output
from .gff_preprocess import gff_parse

primer='CTTGCGGGCGGCGGACTCTCCTCTGAAGATAGAGCGACAGGCAAG'

sub_mat=parasail.matrix_create('AGTC',20,-10)
comp_base_map={'A':'T','T':'A','C':'G','G':'C','[':']', ']':'['}
cigar_map={'M':0, '=':0, 'X':0, 'D':1, 'I':2, 'S':4,'H':4, 'N':3, 'P':5, 'B':5}
cigar_pattern = r'\d+[A-Za-z]'

strand_map={'+':0,'-':1,'+-':2}
inv_strand_map={0:'+',1:'-', 2:'+-'}
strand_switch={'+':'-', '-':'+'}

def revcomp(s):
    return ''.join(comp_base_map[x] for x in s[::-1])

def get_sw(seq):
    global primer
    global sub_mat
    cigar=parasail.sw_trace(seq, primer, 9, 1, sub_mat).cigar
    cigar_op=[(x & 0xf, x >> 4) for x in cigar.seq] #op count
    matches=sum(x[1] for x in cigar_op if x[0]==7)
    total_len=sum(x[1] for i,x in enumerate(cigar_op) if ~((i==0 or i==len(cigar_op)-1) and x[0]==1))
    pid=matches/total_len
    
    return pid

def get_longest_subsequence(seq):
    counts=[(x, sum(1 for _ in y)) for x,y in itertools.groupby(seq) if x=='A' or x=='T']+[('A',0), ('T',0)]
    max_T=max([x for x in counts if x[0]=='T'], key=lambda x:x[1])
    max_A=max([x for x in counts if x[0]=='A'], key=lambda x:x[1])
    return max_T, max_A

def get_blocks(cigar_tuples, ref_start):
    #cigar_map={'M':0, '=':0, 'X':0, 'D':1, 'I':2, 'S':4,'H':4, 'N':3, 'P':5, 'B':5}
    blocks=[]
    rlen=0
    read_coord=1
    ref_coord=ref_start
    
    prev_read_coord=read_coord
    prev_ref_coord=ref_coord
    
    for i in range(len(cigar_tuples)):
        len_op, op= cigar_tuples[i,0], cigar_tuples[i,1]
        
        if op==0:
            read_coord+=len_op
            ref_coord+=len_op
            rlen+=len_op
            
        elif op==1:
            ref_coord+=len_op
            
        elif op==2:
            read_coord+=len_op
            rlen+=len_op
        elif op==3:
            #1-based ref coord
            blocks.append([prev_read_coord, read_coord-1, prev_ref_coord+1, ref_coord])
            
            prev_read_coord=read_coord
            ref_coord+=len_op
            prev_ref_coord=ref_coord
            
        elif op==4:
            rlen+=len_op
            if read_coord==1:
                read_coord+=len_op
                prev_read_coord=read_coord
            else:
                #1-based ref-coord
                blocks.append([prev_read_coord, read_coord-1, prev_ref_coord+1, ref_coord])
    if op==0:
        blocks.append([prev_read_coord, read_coord-1, prev_ref_coord+1, ref_coord])
    return np.array(blocks), ref_coord, rlen

def get_read_info(cigar, flag, ref_pos, seq, check_strand):
    global cigar_pattern
    
    mrna_strand_num=2
    
    if check_strand:
        pid=get_sw(seq)
        rev_pid=get_sw(revcomp(seq))
        best_T, best_A = get_longest_subsequence(seq)

        if pid>rev_pid+0.5 or best_A[1]>best_T[1]+5:
            mrna_strand_num=0

        elif pid+0.5<rev_pid or 5+best_A[1]<best_T[1]:
            mrna_strand_num=1        

    cigar_tuples = np.array([(int(x[:-1]), cigar_map[x[-1]]) for x in re.findall(cigar_pattern, cigar)])
    blocks, ref_coord, rlen=get_blocks(cigar_tuples, int(ref_pos))
    
    s=np.full(len(blocks), ref_pos, dtype=int)
    e=np.full(len(blocks), ref_coord, dtype=int)
    
    
    rev=int(flag)&16

    if rev:
        blocks[:,:2]=np.flip(rlen-blocks[:,:2],axis=1)
        ref_range=np.vstack([e,s]).T
        blocks=np.hstack([blocks, np.flip(blocks[:,2:],axis=1), np.ones((blocks.shape[0],1))])
    else:
        ref_range=np.vstack([s,e]).T
        blocks=np.hstack([blocks, blocks[:,2:], np.zeros((blocks.shape[0],1))])
        
    return blocks, ref_range, mrna_strand_num

def seg_olp(x,y):
    (a,b)= (x,y) if x[0]<y[0] else (y,x)
    return max(0,a[1]-b[0])

def detect_GF_blocks(df_array, final_blocks, l_idx, r_idx, col_map):
    #determine the best exons for each gene
    #calculate boundary distances for each exon
    starts=np.abs(final_blocks[l_idx,2]-df_array[r_idx, col_map['start']])
    ends=np.abs(final_blocks[l_idx,3]-df_array[r_idx, col_map['end']])
    dist=np.stack([starts, ends]).T
    min_dist=starts+ends

    #Ignore read start/end boundary distances
    min_idx=np.where(l_idx==np.min(l_idx))[0]
    min_dist[min_idx]=dist[min_idx,1-final_blocks[np.min(l_idx)][6]]
    max_idx=np.where(l_idx==np.max(l_idx))[0]
    min_dist[max_idx]=dist[max_idx,final_blocks[np.max(l_idx)][6]]

    cluster_gene_exon=npi.group_by(list(zip(l_idx,df_array[r_idx,5])))
    split_gene_exon=cluster_gene_exon.split(np.arange(len(l_idx)))

    #keep the exons with smallest distance
    keep_exons=np.sort(np.array([s[np.argmin(min_dist[s])] for s in split_gene_exon]))
    unique_exon_r_idx=r_idx[keep_exons]
    unique_exon_l_idx=l_idx[keep_exons]
    min_dist=min_dist[keep_exons]

    #determine the best genes for the read by coverage

    #segments covered by a single gene shouldn't contribute to distances
    _,uid,cnts=np.unique(unique_exon_l_idx,return_counts=True, return_index=True)
    singletons=uid[cnts<2]
    min_dist[singletons]=0

    # calculate the length of exons of each gene covered by the read (probably should do inner distance)
    exon_len=np.abs(final_blocks[unique_exon_l_idx,2]-final_blocks[unique_exon_l_idx,3])+1

    #cluster segments by gene
    cluster_gene=npi.group_by(df_array[unique_exon_r_idx,5])
    split_gene=cluster_gene.split(np.arange(len(unique_exon_l_idx)))
    split_gene={df_array[unique_exon_r_idx,5][x[0]]:(x,unique_exon_l_idx[x]) for x in split_gene}

    #sort genes by how many segments of read they cover
    longest_gene=sorted(split_gene.items(), key=lambda x:len(x[1][1]), reverse=True)[0]

    #if a gene covers all segments of a reads, then that gene represents the read
    if np.array_equal(longest_gene[1][1], np.unique(unique_exon_l_idx)):
        keep_genes={longest_gene[0]}

    #otherwise determine a set of genes to keep
    #for each read segment, determine the best gene to keep and then combine
    else:
        #calculate the mean boundary distance and total length of exons covered
        gene_dist={x:(np.mean(min_dist[y[0]]), np.sum(exon_len[y[0]])) for x,y in split_gene.items()}

        cluster_exon=npi.group_by(unique_exon_l_idx)
        split_exon=cluster_exon.split(np.arange(len(unique_exon_l_idx)))

        #get the best gene based on smaller dist, higher covered length, then by name to break ties
        keep_genes=set(sorted([(z,*gene_dist[z]) for z in df_array[unique_exon_r_idx,5][x]], key=lambda g:(g[1],1/g[2],g[0]))[0][0] for x in split_exon)
    keep_genes_idx=np.sort(np.hstack([split_gene[x][0] for x in keep_genes]))

    final_exon_r_idx=unique_exon_r_idx[keep_genes_idx]
    final_exon_l_idx=unique_exon_l_idx[keep_genes_idx]
    
    return final_exon_l_idx, final_exon_r_idx

def get_exon_overlap(read_info, ncls, df_array, col_map, chrom_map, gene_strand_map, trans_exon_counts_map, get_gene_exons, check_strand, gf_only):
    chrom_list=[]
    strand_list=[]
    blocks_list=[]
    ref_range_list=[]
    orientation_list=[]
    gf_output=[]
    mapq_list=[]
    
    block_start_list=[]
    block_end_list=[]
    seq=""
    for read in read_info:
        read_name, ref_name, cigar, flag, ref_pos, read_seq, mapq = read
        if int(flag)&0x900==0:
            seq=read_seq if int(flag)&16==0 else revcomp(read_seq)
        orientation_list.append(int(flag)&16)
        blocks, ref_range, mrna_strand_num=get_read_info(cigar, flag, ref_pos, read_seq, check_strand)
        ref_range_list.append(ref_range)
        blocks_list.append(np.array(blocks))
        chrom_list.append(np.full(len(blocks), chrom_map[ref_name], dtype=int))
        strand_list.append(np.full(len(blocks), mrna_strand_num, dtype=int))
        mapq_list.append(np.full(len(blocks), mapq, dtype=int))
        block_start_list.append(np.full(len(blocks), np.min(blocks[:,:2]), dtype=int))
        block_end_list.append(np.full(len(blocks), np.max(blocks[:,:2]), dtype=int))
        
    if len(blocks_list)==0 or len(seq)==0:
        return [[0, ()]]

    
    ref_range_list=np.vstack(ref_range_list).astype(int)
    merged_blocks_list=np.vstack(blocks_list).astype(int)
    chrom_col=np.hstack(chrom_list)
    strand_col=np.hstack(strand_list)
    mapq_col=np.hstack(mapq_list)
    block_start_list=np.hstack(block_start_list)
    block_end_list=np.hstack(block_end_list)
    
    start_col=(1e10*chrom_col+1e9*strand_col+merged_blocks_list[:,2]).astype(int)
    end_col=(1e10*chrom_col+1e9*strand_col+merged_blocks_list[:,3]).astype(int)

    #q_start, q_end, r_start, r_end, r_start_5prime, r_end_3prime, original_strand, chrom_col, strand_col, start_col, end_col, ref_range_list_start, ref_range_list_end
    final_blocks=np.hstack([merged_blocks_list,np.stack([chrom_col, strand_col, start_col, end_col, mapq_col]).T, ref_range_list, np.stack([block_start_list, block_end_list]).T])
    final_blocks=final_blocks[np.sum(final_blocks[:, :2],axis=1).argsort(kind='mergesort')]
    
    tmp=final_blocks[:,[0,1]]
    tmp_len=tmp[:,1]-tmp[:,0]
    segments=[]
    curr_idx=0
    current=tmp[curr_idx]

    for i in range(1, len(tmp)):
        olp=seg_olp(tmp[i], tmp[curr_idx])
        if olp>5 and max(olp/tmp_len[i], olp/tmp_len[curr_idx])>0.8:
            if tmp_len[i]<tmp_len[curr_idx]:
                continue
            else:
                curr_idx=i
        else:
            segments.append(curr_idx)
            curr_idx=i

    segments.append(curr_idx)
    final_blocks=final_blocks[segments]
    final_blocks=final_blocks[np.abs(final_blocks[:,2]-final_blocks[:,3])>=5]
    
    l_idx, r_idx=ncls.all_overlaps_both(final_blocks[:,9].copy(order='C'), final_blocks[:,10].copy(order='C'), np.arange(len(final_blocks)))
    l_idx_sort=l_idx.argsort()
    l_idx=l_idx[l_idx_sort]
    r_idx=r_idx[l_idx_sort]
    
    if len(l_idx)==0:
        return [[False, ()]]
    
    gf_l_idx, gf_r_idx=detect_GF_blocks(df_array, final_blocks, l_idx, r_idx, col_map)
    
    genes_col=df_array[gf_r_idx, col_map['gene_id']]

    gene_list=tuple(sorted(np.unique(genes_col)))    
    
    #DISABLE Exon Skipping?
    if len(gene_list)<1:
        return [[0, ()]]
    elif len(gene_list)==1 and not gf_only:
        gene_id=gene_list[0]
        try:
            gene_strand=gene_strand_map[gene_id]
        except KeyError:
            return [[0, ()]]
        
        orientation_list=set(orientation_list)
        if len(orientation_list)>1:
            return [[0, ()]]
        else:
            fwd=next(iter(orientation_list))==0
            read_strand="+" if fwd else "-"
            
        try:
            exon_transcript_info=check_exons(final_blocks, df_array, r_idx, l_idx, gene_id, col_map, trans_exon_counts_map, get_gene_exons, read_strand, gene_strand)
        except IndexError:
            print(read_name)
            return [[0, ()]]
        
        if exon_transcript_info[0]:
            return [[0, ()]]
        else:
            return [[1, (gene_id, *exon_transcript_info, read_name)]]
            
    else:
        for bp_exon in np.where(genes_col[1:]!=genes_col[:-1])[0]:
            bp1=final_blocks[gf_l_idx[bp_exon],[7,5]]
            bp2=final_blocks[gf_l_idx[bp_exon+1],[7,4]]
            str_1=final_blocks[gf_l_idx[bp_exon], 6]
            str_2=final_blocks[gf_l_idx[bp_exon+1], 6]

            rbp1=final_blocks[gf_l_idx[bp_exon], [14, 1]]
            rbp2=final_blocks[gf_l_idx[bp_exon+1], [0,15]]
            
            seq1=seq[final_blocks[gf_l_idx[bp_exon],0]:final_blocks[gf_l_idx[bp_exon],1]]
            max1_AT=max(seq1.count('A'), seq1.count('T'))/len(seq1)

            seq2=seq[final_blocks[gf_l_idx[bp_exon+1],0]:final_blocks[gf_l_idx[bp_exon+1],1]]
            max2_AT=max(seq2.count('A'), seq2.count('T'))/len(seq2)
            
            if len(seq1)<20 or len(seq2)<20 or max1_AT>0.75 or max2_AT>0.75:
                return [[0, ()]]
            
            mapq1=final_blocks[gf_l_idx[bp_exon], 11]
            mapq2=final_blocks[gf_l_idx[bp_exon+1], 11]
            
            tid_col=df_array[gf_r_idx, col_map['transcript_id']]
            bp1_intron=True if  tid_col[bp_exon]==-1 else False
            bp2_intron=True if  tid_col[bp_exon+1]==-1 else False
            
            gf_output.append([2, (read_name, genes_col[bp_exon:bp_exon+2], (rbp1, rbp2), (bp1, bp2), (str_1, str_2), (mapq1, mapq2), (bp1_intron, bp2_intron))])
            
        return gf_output 
    
def process(input_queue, output_queue, trans_output_queue, input_event, df_array, col_map, chrom_map, gene_strand_map, trans_exon_counts_map, get_gene_exons, check_strand, gf_only):
    
    
    #define interval tree of exons
    ncls = NCLS64(df_array[:,col_map['new_start']].copy(order='C').astype(int), df_array[:,col_map['new_end']].copy(order='C').astype(int), np.arange(len(df_array)))
        
    while True:
        if (input_queue.empty() and input_event.is_set()):
            break
   
        try:
            read_chunk=input_queue.get(block=False, timeout=10)
            
            for read_info in read_chunk:
                result= get_exon_overlap(read_info, ncls, df_array, col_map, chrom_map, gene_strand_map, trans_exon_counts_map, get_gene_exons, check_strand, gf_only)

                for (status, res) in result:
                    if status==1:
                        trans_output_queue.append(res)
                    elif status==2:
                        output_queue.append(res)
                
        except queue.Empty:
            pass
        
def check_exons(final_blocks, df_array, r_idx, l_idx, gene_id, col_map, trans_exon_counts_map, get_gene_exons, read_strand, gene_strand):
    keep_tr_idx=(df_array[r_idx,col_map['gene_id']]==gene_id) & (df_array[r_idx,col_map['transcript_id']]!=-1)
    if np.sum(keep_tr_idx)<2:
        return [True, None, None]
    
    split_trans_exon=get_gene_exons[gene_id]
    
    #filter out other genes once we have determined the best gene
    tr_r_idx=r_idx[keep_tr_idx]
    tr_l_idx=l_idx[keep_tr_idx]

    if gene_strand!=read_strand:
        final_blocks=np.flip(final_blocks,axis=0)
        tr_l_idx=len(final_blocks)-np.flip(tr_l_idx,axis=0)-1
        tr_r_idx=np.flip(tr_r_idx,axis=0)

    #determine the best exons for each read segment
    #ignore 5' truncation, allow for small differences in distances for best exons
    #if multiple best exons are selected for a segment, create a list of all combos
    starts=np.abs(final_blocks[tr_l_idx,2]-df_array[tr_r_idx, col_map['start']])
    ends=np.abs(final_blocks[tr_l_idx,3]-df_array[tr_r_idx, col_map['end']])

    bp_dist=starts+ends

    cluster_exon=npi.group_by(tr_l_idx)
    split_exon=cluster_exon.split(np.arange(len(tr_l_idx)))

    #ignore 5' truncation
    if gene_strand=="+":
        bp_dist[split_exon[0]]=ends[split_exon[0]]
    else:
        bp_dist[split_exon[0]]=starts[split_exon[0]]

    best_exon_index={tr_l_idx[x][0]:x[bp_dist[x]<=np.min(bp_dist[x])+20] for x in split_exon}
    best_exon_ids={x:tuple(df_array[tr_r_idx, col_map['exon_cid']][y]) for x,y in best_exon_index.items()}
    tmp=np.array([len(x) for x in best_exon_ids.values()])
    
    #if np.prod(tmp)*len(split_trans_exon)>1000000:
    #    print(gene_id, tmp, np.prod(tmp), len(split_trans_exon), np.prod(tmp)*len(split_trans_exon))
    #    return [True, [], []]
    
    exon_combo=[(set(x), ','.join(x)) for x in itertools.product(*list(best_exon_ids.values()))]
    
    #determine which transcript best represents the read
    #get list of exons for each transcript
    
    
    
    
    trans_dist=[(transcript_data[0], exon_list[1] in transcript_data[1]['exons_string'], len(exon_list[0] & set(transcript_data[1]['exons'])), len(exon_list[0]-set(transcript_data[1]['exons']))) for exon_list, transcript_data in itertools.product(exon_combo, split_trans_exon.items())]
    match=sum([x[1] for x in trans_dist])
    closest_transcript=sorted(trans_dist, key=lambda x:x[2], reverse=True)[0]
    best_exon_ids_final=tuple([next(iter(set(v)&set(split_trans_exon[closest_transcript[0]]))) if set(v)&set(split_trans_exon[closest_transcript[0]]) else v[0] for v in best_exon_ids.values()])
    return [match>0, best_exon_ids_final, closest_transcript]

def call_manager(args):
    t=time.time()
    bam_path=args.bam
    gff_path=args.gff
    non_coding_path=args.unannotated
    output_path=args.output
    
    distance_threshold=args.distance_threshold
    min_support=args.min_support
    check_strand=args.check_strand
    include_unannotated=False if non_coding_path==None else True
    gf_only=args.gf_only
    
    df, chrom_map, inv_chrom_map, merged_exon_df, exon_array, col_map,\
gene_df, gene_id_to_name, gene_strand_map, gene_chrom_map, overlapping_genes, trans_exon_counts_map, get_gene_exons = gff_parse(gff_path, non_coding_path, include_unannotated=include_unannotated)
    
    print("Finished reading GFF file.")
    
    bam=pysam.AlignmentFile(bam_path, 'rb')
    
    pmanager = mp.Manager()
    input_queue = pmanager.Queue()
    output_queue = pmanager.list()
    trans_output_queue = pmanager.list()
    input_event=pmanager.Event()

    threads=args.threads
    handlers=[]
    for hid in range(threads):
        p = mp.Process(target=process, args=(input_queue, output_queue, trans_output_queue, input_event, exon_array, col_map, chrom_map, gene_strand_map, trans_exon_counts_map, get_gene_exons, check_strand, gf_only))
        p.start();
        handlers.append(p);

    read_chunk=[]
    read_info=[]
    curr_rname=''

    for k,read in enumerate(bam.fetch(until_eof=True)):
        if k%100000==0:
            print(k, input_queue.qsize(), time.time()-t)

        if input_queue.qsize()>1000:
            #pass
            time.sleep(2)

        if read.qname!=curr_rname and k>1:
            read_chunk.append(read_info)
            read_info=[]
            curr_rname=read.qname

            if len(read_chunk)>=100:
                input_queue.put(read_chunk)
                read_chunk=[]

        #if not (read.is_secondary or read.is_unmapped or read.reference_name not in chrom_map):
        if  read.is_unmapped or read.reference_name not in chrom_map:
            continue
        else:
            if (not read.is_secondary):# or read.reference_name=='chrX':
                read_info.append([read.qname, read.reference_name, read.cigarstring, read.flag, read.reference_start, read.seq, read.mapq])

    if len(read_info)>0:
        read_chunk.append(read_info)

    if len(read_chunk)>0:
        input_queue.put(read_chunk)

    print('input finished')
    input_event.set()
    
    while not input_queue.empty():
        time.sleep(20)
        print(k, input_queue.qsize(), time.time()-t)

    for job in handlers:
        job.join()

    print('read processing finished')
    
    total_output, output=parse_GF_output(output_queue, gene_id_to_name, gene_strand_map, inv_chrom_map, inv_strand_map, overlapping_genes)
    
    print("finished parsing GFs")
    
    with open(os.path.join(output_path, args.prefix+'.read_level.pickle'), 'wb') as handle:
        print(os.path.join(output_path, args.prefix+'.read_level.pickle'))
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return total_output, output, gene_id_to_name
                       
      
                       