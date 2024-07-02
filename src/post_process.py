import numpy.lib.recfunctions as rfn
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import numpy_indexed as npi

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
        rname, init_gene_list, rbp, bp, strands, mapq, bp_intron=x

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
            output[gene_list]={'reads':[], 'info':[],'genes_overlap':genes_overlap, 'is_consistent': gene_list[2]=='consistent'}
        output[gene_list]['reads'].append(rname)
        output[gene_list]['info'].append((gene_1_data, gene_2_data))
        
    
    return total_output, output

def get_cluster(read1, read2, read_names, cluster_type, genes_overlap, consistent, gene_id_to_name, distance_threshold=2, min_support=1):
    
    if cluster_type=="both":
        data_to_fit=np.vstack([read1['bp'], read2['bp']]).T
    elif cluster_type=="left":
        data_to_fit=read1['bp'][:,np.newaxis]
    elif cluster_type=="right":
        data_to_fit=read2['bp'][:,np.newaxis]

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

            r1len =np.max(read1_cluster['read_bp_3p']-read1_cluster['read_bp_5p'])
            r2len =np.max(read2_cluster['read_bp_3p']-read2_cluster['read_bp_5p'])
            
            intron1="Intronic" if np.mean(read1_cluster['intron'])>=0.5 else "Exonic"
            intron2="Intronic" if np.mean(read2_cluster['intron'])>=0.5 else "Exonic"
            
            chrom_1=read1_cluster['chrom'][0]
            chrom_2=read2_cluster['chrom'][0]
            gene_1=read1_cluster['gene_id'][0]
            gene_2=read2_cluster['gene_id'][0]
            readthrough=True if chrom_1==chrom_2 and abs(bp1-bp2)<200000 else False
            
            annotated=(gene_1 in gene_id_to_name) + (gene_2 in gene_id_to_name)
            value={'median_breakpoint_1':(chrom_1, bp1, gene_1, bp1_range, mapq1, r1len, intron1), \
                                              'median_breakpoint_2':(chrom_2, bp2, gene_2, bp2_range, mapq2, r2len, intron2), \
                                              'read_support': read_support,'annotated':annotated, \
                                              'read_names': cluster_read_names,\
                                               'read1_cluster':read1_cluster, 'read2_cluster':read2_cluster,\
                                              'genes_overlap':genes_overlap, 'consistent':consistent, 'readthrough': readthrough}

            cluster_dict[(gene_1, gene_2,'GF{}_{}'.format('' if consistent else '_inconsistent',i))]=value
            i+=1
    return cluster_dict.copy()
    
def get_GFs(output, gene_id_to_name, distance_threshold=20, min_support=1):
    final_gf_double_bp={}
    final_gf_single_bp={}
    for key in output.keys():
        #dtype=[('read_bp_5p', int), ('read_bp_3p', int), ('read_strand', 'O'), ('chrom', 'O'), ('bp', int), ('gene_id','O'), ('gene_name', 'O'), ('gene_strand', 'O'), ('mapq',int), ('intron', bool)])
        read1=np.array([x[0] for x in output[key]['info']], dtype=[('read_bp_5p', int), ('read_bp_3p', int), ('read_strand', 'O'), ('chrom', 'O'), ('bp', int), ('gene_id','O'), ('gene_name', 'O'), ('gene_strand', 'O'), ('mapq',int), ('intron', bool)])
        read2=np.array([x[1] for x in output[key]['info']], dtype=[('read_bp_5p', int), ('read_bp_3p', int), ('read_strand', 'O'), ('chrom', 'O'), ('bp', int), ('gene_id','O'), ('gene_name', 'O'), ('gene_strand', 'O'), ('mapq',int), ('intron', bool)])
        
        read_names=np.array(output[key]['reads'])
        genes_overlap, consistent=output[key]['genes_overlap'],output[key]['is_consistent']
        
        double_bp=get_cluster(read1, read2, read_names, "both", genes_overlap, consistent, gene_id_to_name, distance_threshold, min_support)
        final_gf_double_bp.update(double_bp)
        left_dict=get_cluster(read1, read2, read_names, "left", genes_overlap, consistent, gene_id_to_name, distance_threshold, min_support)
        right_dict=get_cluster(read1, read2, read_names, "right", genes_overlap, consistent, gene_id_to_name, distance_threshold, min_support)
        
        if len(left_dict)<len(right_dict):
            final_gf_single_bp.update(left_dict)
        else:
            final_gf_single_bp.update(right_dict)
            
    return final_gf_double_bp, final_gf_single_bp