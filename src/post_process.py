import numpy.lib.recfunctions as rfn
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import numpy_indexed as npi
import re, pysam, random
import multiprocessing as mp
import parasail
from .utils import *

sub_mat=parasail.matrix_create('AGTC',10,-10)
cigar_map={'M':0, '=':0, 'X':0, 'D':1, 'I':2, 'S':4,'H':4, 'N':3, 'P':5, 'B':5}
cigar_pattern = r'\d+[A-Za-z]'

def get_best_transcript(seq, ref_list):
    global sub_mat
    rev_seq=revcomp(seq)
    
    res=[(i, parasail.sw_striped_sat(s1=s, s2=ref, open=40, extend=10, matrix=sub_mat).score) for s in [seq, rev_seq] for i, ref in ref_list.items()]
    
    res=sorted(res, key=lambda x:x[1], reverse=True)
    best_ref, best_score=res[0]    
    second_best=res[1][1] if len(res)>1 else 1
    
    mapq=min(60, 40*np.log(best_score)*(1-second_best/best_score))
    
    return best_ref, mapq
    
def get_longest_match(seq, ref):
    global sub_mat
    alignment=parasail.sw_trace_striped_sat(s1=seq, s2=ref, open=40, extend=10, matrix=sub_mat)
    matches=[x >> 4 for x in alignment.cigar.seq if x & 0xf ==7]
    max_len=max(matches)
    sum_match=sum(matches)
    return alignment.score, max_len, sum_match

def get_pval(seq, ref_list,n=100):
    global sub_mat
    rev_seq=revcomp(seq)
    
    idx, (score, max_len, sum_match)=max([(i,get_longest_match(seq, ref)) for i, ref in ref_list.items()], key=lambda x:x[1])
    
    rev_idx, (score_rev, max_len_rev, sum_match_rev)=max([(i,get_longest_match(rev_seq, ref)) for i, ref in ref_list.items()], key=lambda x:x[1])
    
    best_idx, best_score, seq, best_max_len, best_sum_match= (idx, score, seq, max_len, sum_match) if score >score_rev else (rev_idx, score_rev, rev_seq, max_len_rev, sum_match_rev)

    ref=ref_list[best_idx]
    
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
    read1, read2, read_names, genes_overlap, consistent, same_segment, gene1_ref_seq, gene2_ref_seq, read_seqs, distance_threshold, min_support, non_coding_gene_id=args
    
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
            
            intron1="Cluster" if gene_1 in non_coding_gene_id else intron1
            intron2="Cluster" if gene_2 in non_coding_gene_id else intron2
            
            best_read1_idx=np.argmax(read1_cluster['read_bp_3p']-read1_cluster['read_bp_5p'])
            best_read2_idx=np.argmax(read2_cluster['read_bp_3p']-read2_cluster['read_bp_5p'])
                        
            read1_seq=read_seqs[cluster_indices][best_read1_idx][read1_cluster['read_bp_5p'][best_read1_idx]-1:read1_cluster['read_bp_3p'][best_read1_idx]]
            read2_seq=read_seqs[cluster_indices][best_read2_idx][read2_cluster['read_bp_5p'][best_read2_idx]-1:read2_cluster['read_bp_3p'][best_read2_idx]]
            
            # get best transcript for each gene
            gene1_best_transcript=get_best_transcript(read1_seq, gene1_ref_seq) if len(gene1_ref_seq)>0 else ('NA',0)
            gene2_best_transcript=get_best_transcript(read2_seq, gene2_ref_seq) if len(gene2_ref_seq)>0 else ('NA',0)
            
            #best_sum_match, best_max_len, best_score, np.mean(tests>=best_score), (best_score-np.mean(tests))/np.std(tests)
            gene1_sig=get_pval(read2_seq, gene1_ref_seq,n=100) if len(gene1_ref_seq)>0 else [-1,-1,-1,-1,-1]
            gene1_sig={'num_matches':gene1_sig[0], 'max_match':gene1_sig[1], 'score':gene1_sig[2], 'pval':gene1_sig[3], 'zscore':gene1_sig[4]}
            gene2_sig=get_pval(read1_seq,gene2_ref_seq,n=100)  if len(gene2_ref_seq)>0 else [-1,-1,-1,-1,-1]
            gene2_sig={'num_matches':gene2_sig[0], 'max_match':gene2_sig[1], 'score':gene2_sig[2], 'pval':gene2_sig[3], 'zscore':gene2_sig[4]}
            
            readthrough=True if chrom_1==chrom_2 and abs(bp1-bp2)<200000 and same_segment else False
            
            value={'median_breakpoint_1':(read1_cluster[0][6], gene_1, chrom_1, bp1, bp1_range, mapq1, r1len, intron1), \
                   'gene1_best_transcript':gene1_best_transcript,\
                   'median_breakpoint_2':(read2_cluster[0][6], gene_2, chrom_2, bp2 , bp2_range, mapq2, r2len, intron2), \
                   'gene2_best_transcript':gene2_best_transcript,\
                   'read_support': read_support, \
                   'read_names': cluster_read_names,\
                   'read1_cluster':read1_cluster, 'read2_cluster':read2_cluster,\
                   'genes_overlap':genes_overlap, 'consistent':consistent, 'readthrough': readthrough,
                   'gene1_sig':gene1_sig, 'gene2_sig':gene2_sig}

            cluster_dict[(gene_1, gene_2,'GF{}_{}'.format('' if consistent else '_inconsistent',i))]=value
            i+=1
    return cluster_dict.copy()

    
def get_GFs(output, gene_transcript_map, tr_file_path, gene_df,non_coding_gene_id, distance_threshold=20, min_support=1, threads=1):
    
    tr_file=pysam.FastaFile(tr_file_path)
    all_tr_seq={re.findall(r"([^.]+)", x.split('|')[0])[0]:tr_file.fetch(x) for x in tr_file.references}
    
    final_gf_double_bp={}
    
    with mp.Pool(threads) as pool:
        input_list=[]

        for key in output.keys():
            #dtype=[('read_bp_5p', int), ('read_bp_3p', int), ('read_strand', 'O'), ('chrom', 'O'), ('bp', int), ('gene_id','O'), ('gene_name', 'O'), ('gene_strand', 'O'), ('mapq',int), ('intron', bool)])
            read1=np.array([x[0] for x in output[key]['info']], dtype=[('read_bp_5p', int), ('read_bp_3p', int), ('read_strand', 'O'), ('chrom', 'O'), ('bp', int), ('gene_id','O'), ('gene_name', 'O'), ('gene_strand', 'O'), ('mapq',int), ('intron', bool)])
            read2=np.array([x[1] for x in output[key]['info']], dtype=[('read_bp_5p', int), ('read_bp_3p', int), ('read_strand', 'O'), ('chrom', 'O'), ('bp', int), ('gene_id','O'), ('gene_name', 'O'), ('gene_strand', 'O'), ('mapq',int), ('intron', bool)])
            
            gene1_id=read1['gene_id'][0]
            gene2_id=read2['gene_id'][0]
            gene1_ref_seq={x:all_tr_seq[x] for x in gene_transcript_map[gene1_id]} if gene1_id in gene_transcript_map else {}
            gene2_ref_seq={x:all_tr_seq[x] for x in gene_transcript_map[gene2_id]} if gene2_id in gene_transcript_map else {}
            
            read_names=np.array(output[key]['reads'])
            read_seqs=np.array(output[key]['read_seqs'])
            
            genes_overlap, consistent, same_segment=output[key]['genes_overlap'], output[key]['is_consistent'], output[key]['same_segment']
            input_list.append([read1, read2, read_names, genes_overlap, consistent, same_segment, gene1_ref_seq, gene2_ref_seq, read_seqs, distance_threshold, min_support, non_coding_gene_id])

        results=pool.imap_unordered(get_cluster, input_list)

        for double_bp in results:
            final_gf_double_bp.update(double_bp)
            
    return final_gf_double_bp


def process_isoforms(bam, gene_strand_map, raw_exons, get_gene_exons, tr_file_path):
    
    tr_file=pysam.FastaFile(tr_file_path)
    all_tr_seq={re.findall(r"([^.]+)", x.split('|')[0])[0]:tr_file.fetch(x) for x in tr_file.references}
    
    #gene:{transcript: {reads, exons} }
    per_read_data={}
    exon_skip={}
    for x in raw_exons:
        
        per_read_data[x[4]]=x
    
        if x[0] not in exon_skip:
            exon_skip[x[0]]={}


        if x[3][0] not in exon_skip[x[0]]:
            exon_skip[x[0]][x[3][0]]={'reads':[], 'exons':[]}

        exon_skip[x[0]][x[3][0]]['exons'].append(x[2])
        exon_skip[x[0]][x[3][0]]['reads'].append(x[4])



    combined_exons={}
    for k,trs in exon_skip.items():
        combined_exons[k]={}

        for tr,skips in trs.items():
            combined_exons[k][tr]={}
            for r,e in sorted(zip(skips['reads'], skips['exons']), key =lambda x:len(x[1]), reverse=True):
                
#                 super_set=get_super_set(e, combined_exons[k][tr].keys())
#                 super_set_rev=get_super_set(e[::-1], combined_exons[k][tr].keys())

#                 if len(super_set)>0:
#                     combined_exons[k][tr][super_set].append(r)

#                 elif len(super_set_rev)>0:
#                     combined_exons[k][tr][super_set_rev].append(r)
#                 else:
#                     combined_exons[k][tr][e]=[r]
                    
                
                if e in combined_exons[k][tr]:
                    combined_exons[k][tr][e].append(r)

                elif e[::-1] in combined_exons[k][tr]:
                    combined_exons[k][tr][e[::-1]].append(r)
                else:
                    combined_exons[k][tr][e]=[r]


    j=0
    reads_to_check=[]
    single_transcript_skips={}
    single_transcript_non_repeat_non_skips={}
    single_transcript_repeat_non_skips={}

    multi_transcript_isoforms={}
    for gene_id, transcripts in combined_exons.items():

        single_transcript_non_repeat_non_skips[gene_id]=[]
        single_transcript_repeat_non_skips[gene_id]=[]
        multi_transcript_isoforms[gene_id]=[]
        #print(gene_id, gene_id_to_name[gene_id])
        gx=get_gene_exons[gene_id]
        k=0
        for transcript_id, exon_list in transcripts.items():
            gx_tr=tuple(gx[transcript_id]['exons'])
            #print(transcript_id)
            for exon_pattern, reads in exon_list.items():
                if len(reads)<5:
                    continue

                if set(exon_pattern).issubset(set(gx_tr)): #all exons are from one transcript
                    gx_inds=[gx_tr.index(x)+1 for x in exon_pattern]
                    non_repeat=len(set(exon_pattern))==len(exon_pattern)

                    # if exons dont repeat
                    if non_repeat:
                        #if exons on reads are in the same order as exons in transcript
                        if (gx_inds==sorted(gx_inds) or gx_inds[::-1]==sorted(gx_inds)): 
                            k+=1
                            if gene_id not in single_transcript_skips:
                                single_transcript_skips[gene_id]=[]
                            single_transcript_skips[gene_id].append((transcript_id, exon_pattern,gx_inds, reads, len(reads)))
                            reads_to_check.append(reads[0])

                        #if exons are in a different order
                        else:
                            k+=1
                            single_transcript_non_repeat_non_skips[gene_id].append((transcript_id, exon_pattern,gx_inds, reads, len(reads)))
                            reads_to_check.append(reads[0])
                            #print('single_transcript non_repeat diff_order')

                    # if exons repeats
                    else:
                        single_transcript_repeat_non_skips[gene_id].append((transcript_id, exon_pattern, gx_inds, reads, len(reads), non_repeat))
                else:
                    gx_inds=[gx_tr.index(x)+1 if x in gx_tr else "NA"  for x in exon_pattern]
                    non_repeat=len(set(exon_pattern))==len(exon_pattern)
                    multi_transcript_isoforms[gene_id].append((transcript_id, exon_pattern, gx_inds, reads, len(reads), non_repeat))
                    reads_to_check.append(reads[0])

        j+=(k>0)

    reads_to_check=set(reads_to_check)

    for gene_id in single_transcript_non_repeat_non_skips.copy():
        if len(single_transcript_non_repeat_non_skips[gene_id])==0:
            del single_transcript_non_repeat_non_skips[gene_id]

    for gene_id in multi_transcript_isoforms.copy():
        if len(multi_transcript_isoforms[gene_id])==0:
            del multi_transcript_isoforms[gene_id]

    for gene_id in single_transcript_repeat_non_skips.copy():
        if len(single_transcript_repeat_non_skips[gene_id])==0:
            del single_transcript_repeat_non_skips[gene_id]
    
    bam=pysam.AlignmentFile(bam, 'rb')

    read_seqs={}
    for k,read in enumerate(bam.fetch(until_eof=True)):
        if read.qname in reads_to_check and (read.is_secondary==False and read.is_supplementary==False):
            read_seqs[read.qname]=(read.seq, read.cigarstring)

    sub_mat=parasail.matrix_create('AGTC',10,-10)

    final_skips={}
    for gene_id, skips in single_transcript_skips.items():
        if len(skips)>0:
            for skip in skips:
                transcript_id, exon_pattern, gx_inds, read_list, read_support=skip
                try:
                    read_name=read_list[0]
                    s,c=read_seqs[read_name]
                    s1=all_tr_seq[transcript_id]
                    s1=s1 if gene_strand_map[gene_id]=="+" else revcomp(s1)
                    cigar_tuples = np.array([(int(x[:-1]), cigar_map[x[-1]]) for x in re.findall(cigar_pattern, c)])
                    left_clip=cigar_tuples[0,0] if cigar_tuples[0,1]==4 else 0
                    right_end=len(s)-cigar_tuples[-1,0] if cigar_tuples[-1,1]==4 else len(s)
                    alignment=parasail.sg_trace_scan_32(s[left_clip:right_end], s1, 10, 1, sub_mat)
                    cigar=alignment.cigar
                    cigar_string=cigar.decode.decode('utf-8')
                    cigar_info=[(cigar.decode_op(x).decode('utf-8'), cigar.decode_len(x)) for x in cigar.seq]
                    max_del=max([x[1] for x in cigar_info[1:-1] if x[0]=='D']+[0])
                    if max_del>30:
                        if gene_id not in final_skips:
                            final_skips[gene_id]=[]
                        final_skips[gene_id].append(skip)
                    else:
                        pass#print("fail", gene_id_to_name[gene_id], gene_id, transcript_id, len(read_list))
                except KeyError as e:
                    #print(repr(e))
                    pass

    return final_skips, single_transcript_skips, single_transcript_non_repeat_non_skips, single_transcript_repeat_non_skips, multi_transcript_isoforms, reads_to_check
