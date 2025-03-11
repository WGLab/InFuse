import re, os, argparse, datetime

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

from .utils import *

cigar_map={'M':0, '=':0, 'X':0, 'D':1, 'I':2, 'S':4,'H':4, 'N':3, 'P':5, 'B':5}
cigar_pattern = r'\d+[A-Za-z]'

strand_map={'+':0,'-':1,'+-':2}
inv_strand_map={0:'+',1:'-', 2:'+-'}
strand_switch={'+':'-', '-':'+'}

block_col_map={'q_start': 0,
 'q_end': 1,
 'r_start': 2,
 'r_end': 3,
 'r_start_5prime': 4,
 'r_end_3prime': 5,
 'original_strand': 6,
 'chrom_col': 7,
 'strand_col': 8,
 'start_col': 9,
 'end_col': 10,
 'mapq_col': 11,
 'ref_range_list_start': 12,
 'ref_range_list_end': 13,
 'alignment_start_list': 14,
 'alignment_end_list': 15,
 'seg_num_col': 16,
 'start_end_mask':17}

def get_blocks(cigar_tuples, ref_start):
    # input is cigar tuple and reference start coordinate
    # output is list of alignment blocks, reference start and end coordinates
    # for the whole read after trimming bad quality ends, and read length
    # blocks are in the format [read_start, read_end, ref_start, ref_end]
    # block list is sorted in the order of reference plus strand coordinates
    
    # cigar_map={'M':0, '=':0, 'X':0, 'D':1, 'I':2, 'S':4,'H':4, 'N':3, 'P':5, 'B':5}
    blocks=[]
    rlen=0
    read_coord=1
    ref_coord=ref_start
    
    prev_read_coord=read_coord
    prev_ref_coord=ref_coord
    
    block_matches=0
    block_len=0
    pid=[]
    
    for i in range(len(cigar_tuples)):
        len_op, op= cigar_tuples[i,0], cigar_tuples[i,1]
        
        if op==0:
            read_coord+=len_op
            ref_coord+=len_op
            rlen+=len_op
            block_matches+=len_op
            block_len+=len_op
            
        elif op==1:
            ref_coord+=len_op
            block_len+=len_op
            
        elif op==2:
            read_coord+=len_op
            rlen+=len_op
            block_len+=len_op
            
        elif op==3:
            #1-based ref coord
            pid.append([block_matches, block_len])
            blocks.append([prev_read_coord, read_coord-1, prev_ref_coord+1, ref_coord])
            
            block_matches, block_len=0, 0
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
                pid.append([block_matches, block_len])
                blocks.append([prev_read_coord, read_coord-1, prev_ref_coord+1, ref_coord])
                    
                block_matches, block_len=0, 0
    if op==0:
        pid.append([block_matches, block_len])
        blocks.append([prev_read_coord, read_coord-1, prev_ref_coord+1, ref_coord])
    
    pid=np.array(pid)
    blocks=np.array(blocks)
    
    #remove any blocks with less than 50% matches if it is the starting or ending block
    start_end_mask=np.full(len(pid), False, dtype=bool)
    start_end_mask[[0,-1]]=True
    final_mask=(pid[:,0]/pid[:,1]>0.5)|(~start_end_mask)
    blocks=blocks[final_mask]
    start_end_mask=start_end_mask[final_mask]
    if len(blocks)==0:
        return [], None, None, None, None
    else:
        start, end= blocks[0,2], blocks[-1,3]
        return blocks, start, end, rlen, start_end_mask

def get_read_info(cigar, flag, ref_pos, check_strand):
    # input is cigar string, read flag, reference starting position, and whether
    # the read strand should be checked or not (only in case of dRNA or pychopper oriented reads)
    # output is list of alignment blocks, list of read alignment start and end, and strand number
    # block list is sorted in the order of reference plus strand coordinates
    # block output format is [block read start, block read end, block ref start, block ref end, ...
    # ..., 5' block ref start, 3' block ref end, 0 /1 for +/- read strand]
    
    global cigar_pattern
    
    mrna_strand_num=2
    rev=int(flag)&16
    
    if check_strand:
        mrna_strand_num=1 if rev else 0

    # get cigar tuples from string
    cigar_tuples = np.array([(int(x[:-1]), cigar_map[x[-1]]) for x in re.findall(cigar_pattern, cigar)])
    
    # get  alignment blocks, possibly trimmed, and ref start/end, total read length
    # alignment blocks are sorted in the order of plus strand reference coordinate
    # and the read coordinates obtained are with respect to plus strand
    blocks, start, end, rlen, start_end_mask=get_blocks(cigar_tuples, int(ref_pos))
    
    if len(blocks)==0:
        return [], [], None, None
    s=np.full(len(blocks), start, dtype=int)
    e=np.full(len(blocks), end, dtype=int)  

    # convert read coordinates from cigar reported coordinates to actual 5' to 3' coordinates
    # if the read is mapped to minus strand, then convert and reverse read coordinates
    if rev:
        # convert read coordinates to start from the other end
        blocks[:,:2]=rlen-blocks[:,:2]+1
        
        # switch start end of each block to change the direction
        blocks[:,:2]=np.flip(blocks[:,:2],axis=1)
        
        # create a list of reference start and end with respect to 5' to 3' direction
        ref_range=np.vstack([e,s]).T
        
        # create columns that show reference block start and ends with respect to 5' to 3' direction
        five_to_three_prime_block_ref=np.flip(blocks[:,2:],axis=1)
        
        # add five_to_three_prime_block_ref and a column 1 for original read strand
        blocks=np.hstack([blocks, five_to_three_prime_block_ref, np.ones((blocks.shape[0],1))])
    
    # if the read is mapped to plus strand, then no change is needed
    else:
        # create a list of reference start and end with respect to 5' to 3' direction
        ref_range=np.vstack([s,e]).T
        
        # create columns that show reference block start and ends with respect to 5' to 3' direction
        five_to_three_prime_block_ref=blocks[:,2:]
        
        # add five_to_three_prime_block_ref and a column 0 for original read strand
        blocks=np.hstack([blocks, five_to_three_prime_block_ref, np.zeros((blocks.shape[0],1))])
        
    return blocks, ref_range, mrna_strand_num, start_end_mask

def seg_olp(x,y):
    (a,b)= (x,y) if x[0]<y[0] else (y,x)
    return max(0,a[1]-b[0])

def get_gene_tuples(gene_list):
    return {tuple(sorted([x,y])) for i, x in enumerate(gene_list) for y in gene_list[i+1:]}

def get_gene_tuples_from_prod(list1, list2):
    return {tuple(sorted([x,y])) for x,y in itertools.product(list1, list2) if x!=y}

def detect_GF_blocks(df_array, final_blocks, l_idx, r_idx, col_map):
    global block_col_map
    # Determine the best exons for each gene  

    # remove gene level row if the gene has any exonic overlap with the read
    # (i.e. the read is entirely contained within introns, or anisense strand in case of check_strand=False)
    cluster_gene = npi.group_by(df_array[r_idx, col_map['gene_id']])
    split_gene = cluster_gene.split(np.arange(len(r_idx)))
    keep_exonic_idx=np.concatenate([z[df_array[r_idx[z]][:,col_map['transcript_id']]!=-1] if np.any(df_array[r_idx[z]][:,col_map['transcript_id']]!=-1) else z for z in split_gene])

    r_idx=r_idx[keep_exonic_idx]
    l_idx=l_idx[keep_exonic_idx]

    # remove gene level row if the corresponding alignment block overlaps exon of another gene
    cluster_block = npi.group_by(l_idx)
    split_block = cluster_block.split(np.arange(len(l_idx)))

    keep_exonic_idx=np.concatenate([z[df_array[r_idx[z]][:,col_map['transcript_id']]!=-1] if np.any(df_array[r_idx[z]][:,col_map['transcript_id']]!=-1) else z for z in split_block])
    r_idx=r_idx[keep_exonic_idx]
    l_idx=l_idx[keep_exonic_idx]

    l_idx_sort=l_idx.argsort()
    l_idx=l_idx[l_idx_sort]
    r_idx=r_idx[l_idx_sort]

    # Calculate boundary distances for each exon
    starts = np.abs(final_blocks[l_idx, block_col_map['r_start']] - df_array[r_idx, col_map['start']])
    ends = np.abs(final_blocks[l_idx, block_col_map['r_end']] - df_array[r_idx, col_map['end']])
    dist = np.stack([starts, ends]).T
    min_dist = starts + ends
    
    #Ignore read start/end boundary distances
    min_idx = np.where((l_idx == np.min(l_idx))&(final_blocks[l_idx,block_col_map['start_end_mask']]))[0]
    min_dist[min_idx] = dist[min_idx, 1 - final_blocks[np.min(l_idx), block_col_map['original_strand']]]
    max_idx = np.where((l_idx == np.max(l_idx))&(final_blocks[l_idx,block_col_map['start_end_mask']]))[0]
    min_dist[max_idx] = dist[max_idx, final_blocks[np.max(l_idx), block_col_map['original_strand']]]
    
        
    # Group by gene_id and block
    cluster_gene_exon = npi.group_by(list(zip(l_idx, df_array[r_idx, col_map['gene_id']])))
    split_gene_exon = cluster_gene_exon.split(np.arange(len(l_idx)))

    # Consider all exons of a gene overlapping a given block
    # Keep the exon with the smallest distance for each gene/block pair
    keep_exons = [s[np.argmin(min_dist[s])] for s in split_gene_exon]
    keep_exons = np.sort(np.array(keep_exons))
    unique_exon_r_idx = r_idx[keep_exons]
    unique_exon_l_idx = l_idx[keep_exons]
    min_dist = min_dist[keep_exons]
    
    #segments covered by a single gene shouldn't contribute to distances
    _,uid,cnts=np.unique(unique_exon_l_idx,return_counts=True, return_index=True)
    singletons=uid[cnts<2]
    min_dist[singletons]=0

    ## Determine the best genes for the read
    # Calculate the length of read blocks
    exon_len = np.abs(final_blocks[:, block_col_map['r_start']] - final_blocks[:, block_col_map['r_end']]) + 1

    # Cluster blocks by gene_id
    cluster_gene = npi.group_by(df_array[unique_exon_r_idx, col_map['gene_id']])

    #get indices of l/r_idx that correspond to each gene
    split_gene = cluster_gene.split(np.arange(len(unique_exon_l_idx)))

    # key=gene_id, value= (l/r_idx indices for each gene, l_idx values for each gene)
    split_gene_dict = {df_array[unique_exon_r_idx, col_map['gene_id']][x[0]]: (x, unique_exon_l_idx[x]) for x in split_gene}

    # Cluster exons by l_idx
    cluster_exon = npi.group_by(unique_exon_l_idx)
    split_exon = cluster_exon.split(np.arange(len(unique_exon_l_idx)))

    # get list of all contiguous candidate gene fusion pairs
    # discard any pair that overlaps in the same alignment block
    # 
    gene_tuples=set()
    for k in range(len(split_exon)-1):
        curr_idx=df_array[unique_exon_r_idx[split_exon[k]]][:,col_map['gene_id']]
        next_idx=df_array[unique_exon_r_idx[split_exon[k+1]]][:,col_map['gene_id']]
        pairs=get_gene_tuples_from_prod(curr_idx, next_idx)
        gene_tuples=gene_tuples.union(pairs)

    overlapping_gene_tuples=[get_gene_tuples(df_array[unique_exon_r_idx[l]][:, col_map['gene_id']]) for l in split_exon]
    overlapping_gene_tuples=set.union(*overlapping_gene_tuples)
    candidate_fusions=list(gene_tuples-overlapping_gene_tuples)

    candidate_genes_dist={}
    for x in candidate_fusions:
        temp1, temp2 = split_gene_dict[x[0]][0], split_gene_dict[x[1]][0]
        temp1=[] if (len(temp1)==1 and str(df_array[r_idx[temp1], col_map['transcript_id']][0])=='-1') else temp1
        temp2=[] if (len(temp2)==1 and str(df_array[r_idx[temp2], col_map['transcript_id']][0])=='-1') else temp2
        temp_idx=np.concatenate([split_gene_dict[x[0]][1], split_gene_dict[x[1]][1]])
        candidate_genes_dist[x]=(np.sum(min_dist[temp1]), np.sum(min_dist[temp2]), np.sum(exon_len[np.setdiff1d(np.arange(len(exon_len)), temp_idx)]))

    for x, y in split_gene_dict.items():
            candidate_genes_dist[(x,)]=(np.sum(min_dist[y[0]]), np.sum(exon_len[np.setdiff1d(np.arange(len(exon_len)), y[1])]))

    best_genes=min(list(candidate_genes_dist.keys()), key=lambda x: sum(candidate_genes_dist[x]))

    # Get the indices of the best genes to keep
    keep_genes_idx = np.sort(np.hstack([split_gene_dict[x][0] for x in best_genes]))

    # Final selected exon indices
    final_exon_r_idx = unique_exon_r_idx[keep_genes_idx]
    final_exon_l_idx = unique_exon_l_idx[keep_genes_idx]
    
    return final_exon_l_idx, final_exon_r_idx

def get_alignment_block_indices(read_info, ncls, df_array, chrom_map, col_map, check_strand):
    global block_col_map
    check=False
    chrom_list=[]
    strand_list=[]
    blocks_list=[]
    ref_range_list=[]
    orientation_list=[]
    
    mapq_list=[]
    
    alignment_start_list=[]
    alignment_end_list=[]
    seg_num_list=[]
    start_end_mask_list=[]
    seq=""
    
    # get alignment blocks for each read alignment
    # to do: maybe drop a read if it has more than 2 supplementary alignments
    for seg_num, read in enumerate(read_info):
        read_name, ref_name, cigar, flag, ref_pos, read_seq, mapq = read
        if int(flag)&0x900==0:
            seq=read_seq if int(flag)&16==0 else revcomp(read_seq)
        orientation_list.append(int(flag)&16)
        blocks, ref_range, mrna_strand_num, start_end_mask=get_read_info(cigar, flag, ref_pos, check_strand)
        
        if len(blocks)==0:
            continue
        ref_range_list.append(ref_range)
        blocks_list.append(np.array(blocks))
        chrom_list.append(np.full(len(blocks), chrom_map[ref_name], dtype=int))
        strand_list.append(np.full(len(blocks), mrna_strand_num, dtype=int))
        mapq_list.append(np.full(len(blocks), mapq, dtype=int))
        alignment_start_list.append(np.full(len(blocks), np.min(blocks[:,:2]), dtype=int))
        alignment_end_list.append(np.full(len(blocks), np.max(blocks[:,:2]), dtype=int))
        seg_num_list.append(np.full(len(blocks), seg_num, dtype=int))
        start_end_mask_list.append(start_end_mask)
        
    if len(blocks_list)==0 or len(seq)==0:
        return check, 0, len(blocks_list)==0, len(seq)==0, None, None, None, None, None

    # combine block information across the alignments
    ref_range_list=np.vstack(ref_range_list).astype(int)
    merged_blocks_list=np.vstack(blocks_list).astype(int)
    chrom_col=np.hstack(chrom_list)
    strand_col=np.hstack(strand_list)
    mapq_col=np.hstack(mapq_list)
    alignment_start_list=np.hstack(alignment_start_list)
    alignment_end_list=np.hstack(alignment_end_list)
    seg_num_col=np.hstack(seg_num_list)
    start_end_mask_list=np.hstack(start_end_mask_list).astype(int)
    
    # create linear start and end columns for interval tree
    start_col=(1e10*chrom_col+1e9*strand_col+merged_blocks_list[:,2]).astype(int)
    end_col=(1e10*chrom_col+1e9*strand_col+merged_blocks_list[:,3]).astype(int)

    
    #final block format
    # 0:q_start, 1:q_end, 2:r_start, 3:r_end, 4:r_start_5prime, 5:r_end_3prime, 6:original_strand,
    # 7:chrom_col, 8:strand_col, 9:start_col, 10:end_col, 11:mapq_col, 
    # 12:ref_range_list_start, 13:ref_range_list_end, 14:alignment_start_list, 15: alignment_end_list,
    # 16:seg_num_col, 17:start_end_mask
    final_blocks=np.hstack([merged_blocks_list,np.stack([chrom_col, strand_col, start_col, end_col, mapq_col]).T, ref_range_list, np.stack([alignment_start_list, alignment_end_list, seg_num_col, start_end_mask_list]).T])
    
    #sort by block start and ends, then internally within each block
    final_blocks=final_blocks[np.lexsort((final_blocks[:, block_col_map['q_start']], final_blocks[:, block_col_map['alignment_end_list']], final_blocks[:, block_col_map['alignment_start_list']]))]
    
    
    # if a pair of overlapping blocks from different alignments (based on read coordinates) are found, pick the longer block
    # overlap occurs overlap is longer than 5bp covers at least 80% of either block.
    # to do: are more sophisticated pruning or merging of blocks from different segments, e.g.
    # [a,b,c,d] [e,f,g,h] where c,d overlap with e,f (unlikely but still). this case will not
    # be handled by current method. a better approach would be to go d->a and e->h and check for overlaps
    tmp=final_blocks[:,[block_col_map['q_start'],block_col_map['q_end']]]
    tmp_len=tmp[:,block_col_map['q_end']]-tmp[:,block_col_map['q_start']]+1
    segments=[]
    curr_idx=0

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
    final_blocks=final_blocks[np.abs(final_blocks[:,block_col_map['r_start']]-final_blocks[:,block_col_map['r_end']])>=5]

    l_idx, r_idx=ncls.all_overlaps_both(final_blocks[:,block_col_map['start_col']].copy(order='C'), final_blocks[:,block_col_map['end_col']].copy(order='C'), np.arange(len(final_blocks)))
    l_idx_sort=l_idx.argsort()
    l_idx=l_idx[l_idx_sort]
    r_idx=r_idx[l_idx_sort]
    
    if len(l_idx)==0:
        return check, 1, None, None, None, final_blocks, None, None, None
    
    gf_l_idx, gf_r_idx=detect_GF_blocks(df_array, final_blocks, l_idx, r_idx, col_map)
    
    check=True
    
    # if a single alignment block is aligned to multiple genes, remove the latter one in 5' to 3' direction of read
    unique=np.unique(gf_l_idx, return_index=True)[1]
    gf_l_idx, gf_r_idx=gf_l_idx[unique], gf_r_idx[unique]
    
    return check, gf_l_idx, gf_r_idx, l_idx, r_idx, final_blocks, seq, read_name, orientation_list

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

    #for each alignment block, get indexes of all exons matches that are within 20 dist of the best matching exon
    best_exon_index={tr_l_idx[x][0]:x[bp_dist[x]<=np.min(bp_dist[x])+10] for x in split_exon}
    
    #for each alignment block, get names of all exons matching criterion above
    best_exon_ids={x:tuple(df_array[tr_r_idx, col_map['exon_cid']][y]) for x,y in best_exon_index.items()}
    tmp=np.array([len(x) for x in best_exon_ids.values()])
    
    if np.prod(tmp)*len(split_trans_exon)>1000000:
        #print(gene_id, tmp, np.prod(tmp), len(split_trans_exon), np.prod(tmp)*len(split_trans_exon))
        return [True, [], []]
    
    exon_combo=[(set(x), ','.join(x)) for x in itertools.product(*list(best_exon_ids.values()))]
    
    #determine which transcript best represents the read
    #get list of exons for each transcript
    # transcript name
    # is exon combo string contained in transcript exon string
    # how many exons are in transcript
    # how many exons are not covered by the transcript 
    trans_dist=[(transcript_data[0],\
                 exon_list[1] in transcript_data[1]['exons_string'],\
                 len(exon_list[0] & set(transcript_data[1]['exons'])),\
                 len(exon_list[0]-set(transcript_data[1]['exons'])))\
                 for exon_list, transcript_data in itertools.product(exon_combo, split_trans_exon.items())]
            
    match=sum([x[1] for x in trans_dist])
    closest_transcript=sorted(trans_dist, key=lambda x:x[2], reverse=True)[0]
    
    #for each alignment block, get the exon of the best transcript that the block maps to.
    best_exon_ids_final=tuple([next(iter(set(v)&set(split_trans_exon[closest_transcript[0]]['exons']))) if set(v)&set(split_trans_exon[closest_transcript[0]]['exons']) else v[0] for v in best_exon_ids.values()])
    return [match>0, best_exon_ids_final, closest_transcript]

def get_exon_overlap(read_info, ncls, df_array, col_map, chrom_map, gene_strand_map, trans_exon_counts_map, get_gene_exons, check_strand, gf_only):
    global block_col_map
    
    gf_output=[]
    
    check, gf_l_idx, gf_r_idx, l_idx, r_idx, final_blocks, seq, read_name, orientation_list=get_alignment_block_indices(read_info, ncls, df_array, chrom_map, col_map, check_strand)
    
    if check==False:
        return [[0, ()]]
    
    genes_col=df_array[gf_r_idx, col_map['gene_id']]

    gene_list=tuple(sorted(np.unique(genes_col)))
    
    #DISABLE Exon Skipping?
    if len(gene_list)<1:
        return [[0, ()]]
    elif len(gene_list)==1:
        if gf_only:
            return [[0, ()]]
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
            bp1 = final_blocks[gf_l_idx[bp_exon], [block_col_map['chrom_col'], block_col_map['r_end_3prime']]]
            bp2 = final_blocks[gf_l_idx[bp_exon + 1], [block_col_map['chrom_col'], block_col_map['r_start_5prime']]]
            str_1 = final_blocks[gf_l_idx[bp_exon], block_col_map['original_strand']]
            str_2 = final_blocks[gf_l_idx[bp_exon + 1], block_col_map['original_strand']]

            rbp1 = final_blocks[gf_l_idx[bp_exon], [block_col_map['alignment_start_list'], block_col_map['q_end']]]
            rbp2 = final_blocks[gf_l_idx[bp_exon + 1], [block_col_map['q_start'], block_col_map['alignment_end_list']]]

            seq1 = seq[final_blocks[gf_l_idx[bp_exon], block_col_map['q_start']]:final_blocks[gf_l_idx[bp_exon], block_col_map['q_end']]]
            max1_AT = max(seq1.count('A'), seq1.count('T')) / len(seq1)

            seq2 = seq[final_blocks[gf_l_idx[bp_exon + 1], block_col_map['q_start']]:final_blocks[gf_l_idx[bp_exon + 1], block_col_map['q_end']]]
            max2_AT = max(seq2.count('A'), seq2.count('T')) / len(seq2)

            if len(seq1) < 20 or len(seq2) < 20 or max1_AT > 0.75 or max2_AT > 0.75:
                return [[0, ()]]

            mapq1 = final_blocks[gf_l_idx[bp_exon], block_col_map['mapq_col']]
            mapq2 = final_blocks[gf_l_idx[bp_exon + 1], block_col_map['mapq_col']]

            tid_col = df_array[gf_r_idx, col_map['transcript_id']]
            bp1_intron = True if tid_col[bp_exon] == -1 else False
            bp2_intron = True if tid_col[bp_exon + 1] == -1 else False

            same_segment = True if final_blocks[gf_l_idx[bp_exon], block_col_map['seg_num_col']] == final_blocks[gf_l_idx[bp_exon + 1], block_col_map['seg_num_col']] else False

            gf_output.append([2, (read_name, genes_col[bp_exon:bp_exon + 2], (rbp1, rbp2), (bp1, bp2), (str_1, str_2), (mapq1, mapq2), (bp1_intron, bp2_intron), same_segment, seq)])
            
        return gf_output if len(gf_output)>0 else [[0, ()]]
    
def process(input_queue, output_queue, trans_output_queue, gene_output_queue, input_event, df_array, col_map, chrom_map, gene_strand_map, trans_exon_counts_map, get_gene_exons, check_strand, gf_only):
    
    
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
                        read_name, gene_name=res[-1], [res[0]]
                        gene_output_queue.append((read_name, gene_name))
                    elif status==2:
                        output_queue.append(res)
                        read_name, gene_name=res[0], list(res[1])
                        gene_output_queue.append((read_name, gene_name))
                
        except queue.Empty:
            pass
        

def print_gene_ids(gene_output_queue, path):
    with open(path, 'w') as f:
        for read, gene_name in gene_output_queue:
            for g in gene_name:
                f.write(f'{read}\t{g}\n')
        
def call_manager(args, gff_data=None, cl=True):
    bam_path=args.bam
    gff_path=args.gff
    non_coding_path=args.unannotated
    output_path=args.output
    
    distance_threshold=args.distance_threshold
    min_support=args.min_support
    check_strand=args.check_strand
    include_unannotated=False if non_coding_path==None else True
    gf_only=args.gf_only
    
    if cl:
        df, chrom_map, inv_chrom_map, merged_exon_df, exon_array, col_map,\
gene_df, gene_id_to_name, gene_strand_map, gene_chrom_map, overlapping_genes, trans_exon_counts_map, get_gene_exons, non_coding_gene_id = gff_parse(gff_path, non_coding_path, check_strand, include_unannotated=include_unannotated)
        
    else:
        df, chrom_map, inv_chrom_map, merged_exon_df, exon_array, col_map,\
gene_df, gene_id_to_name, gene_strand_map, gene_chrom_map, overlapping_genes, trans_exon_counts_map, get_gene_exons, non_coding_gene_id = gff_data
    
    print("Finished reading GFF file.", flush=True)
    
    t=time.time()
    bam=pysam.AlignmentFile(bam_path, 'rb')
    
    pmanager = mp.Manager()
    input_queue = pmanager.Queue()
    output_queue = pmanager.list()
    trans_output_queue = pmanager.list()
    gene_output_queue = pmanager.list()
    input_event=pmanager.Event()

    threads=args.threads
    handlers=[]
    for hid in range(threads):
        p = mp.Process(target=process, args=(input_queue, output_queue, trans_output_queue, gene_output_queue, input_event, exon_array, col_map, chrom_map, gene_strand_map, trans_exon_counts_map, get_gene_exons, check_strand, gf_only))
        p.start();
        handlers.append(p);

    bam_data={}    
    for k, read in enumerate(bam.fetch(until_eof=True)):
        if k%500000==0:
            print('{}: Reading BAM File {} {} {}\n'.format(str(datetime.datetime.now()), k, input_queue.qsize(), time.time()-t), flush=True)

        if read.flag&260!=0:
                continue

        if read.qname not in bam_data:
            bam_data[read.qname]=[]

        if read.reference_name in chrom_map:
            bam_data[read.qname].append([read.qname, read.reference_name, read.cigarstring, read.flag, read.reference_start, read.seq, read.mapq])
    
    print('{}: Finished Reading BAM File {} {} {}\n'.format(str(datetime.datetime.now()), k, input_queue.qsize(), time.time()-t), flush=True)
    
    for chunk in split_list(list(bam_data.values()),n=1000):
        input_queue.put(chunk)
        #print('.',end="")
       
    print('{}: Processing reads. Remaining chunks={}  Time elapsed={}s'.format(str(datetime.datetime.now()), input_queue.qsize(), time.time()-t), flush=True)
    input_event.set()

    while not input_queue.empty():
        if input_queue.qsize()>10:
            print('{}: Processing reads. Remaining chunks={}  Time elapsed={}s'.format(str(datetime.datetime.now()), input_queue.qsize(), time.time()-t), flush=True)
            time.sleep(5)
        else:
            break

    for job in handlers:
        job.join()

    print('{}: Read processing finished in={}s'.format(str(datetime.datetime.now()), time.time()-t), flush=True)

    print_gene_ids(gene_output_queue, os.path.join(output_path, args.prefix+'.read_to_gene'))
    total_output, output=parse_GF_output(output_queue, gene_id_to_name, gene_strand_map, inv_chrom_map, inv_strand_map, overlapping_genes)
    
    print("{}: Finished parsing GFs".format(str(datetime.datetime.now())), flush=True)
    
    gene_transcript_map={x:list(y.keys()) for x,y in get_gene_exons.items()}
    raw_exons=[]
    if output_path!=None:
        with open(os.path.join(output_path, args.prefix+'.read_level.pickle'), 'wb') as handle:
            print("{}: Saving intermediate Fusion results in: {}".format(str(datetime.datetime.now()), os.path.join(output_path, args.prefix+'.read_level.pickle')), flush=True)
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if not gf_only:
            for x in trans_output_queue:
                raw_exons.append(x)

            with open(os.path.join(output_path, args.prefix+'.raw_exons.pickle'), 'wb') as handle:
                print("{}: Saving intermediate exon patterns results in: {}".format(str(datetime.datetime.now()), os.path.join(output_path, args.prefix+'.raw_exons.pickle')), flush=True)
                pickle.dump(raw_exons, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return total_output, output, gene_id_to_name, gene_df, gene_transcript_map, gene_strand_map, get_gene_exons, non_coding_gene_id, raw_exons
                       
    else:
        if not gf_only:
            for x in trans_output_queue:
                raw_exons.append(x)
                
        return total_output, output, gene_id_to_name, gene_df, gene_transcript_map, gene_strand_map, get_gene_exons, non_coding_gene_id, raw_exons
                       