import pandas as pd
import numpy as np
from ncls import NCLS64
import numpy_indexed as npi

def gff_parse(gff_path, non_coding_path, check_strand=False, include_unannotated=False):
    strand_map={'+':0,'-':1,'+-':2}
    df=pd.read_csv(gff_path, sep='\t', comment='#', header=None)
    df.rename(columns={0:'chrom', 2:'feature', 3:'start', 4:'end',6:'strand', 8:'info'}, inplace=True)

    chrom_map={x:i+1 for i,x in enumerate(df.chrom.unique())}
    inv_chrom_map={x:y for y,x in chrom_map.items()}

    df['gene_id']=df['info'].str.extract(r'gene_id=([^;]+)')
    df['gene_name']=df['info'].str.extract(r'gene_name=([^;]+)')
    df['transcript_id']=df['info'].str.extract(r'transcript_id=([^;]+)')
    df['exon_id']=df['info'].str.extract(r'exon_id=([^;]+)')
    df['exon_number']=df['info'].str.extract(r'exon_number=([^;]+)')
    df['gene_type']=df['info'].str.extract(r'gene_type=([^;]+)')
    df.drop(columns=[1,5,7,'info'], inplace=True)
    
    df['exon_id']=df['exon_id'].fillna(df['gene_id'])
    df['exon_number']=df['exon_number'].fillna(-1)
    df['transcript_id']=df['transcript_id'].fillna(-1)
    df['exon_number']=pd.to_numeric(df['exon_number'])
    df["exon_cid"]=df['gene_id']+df['chrom']+"|"+df['start'].astype(str)+"|"+df['end'].astype(str)
    
    
    
    if check_strand:
        #add anti-sense strand for each gene
        exon_df=df[(df.feature=='exon')|(df.feature=='gene')]
        anti_sense_exon_gene_df=df[df.feature=='gene'].copy()
        anti_sense_exon_gene_df['strand']=anti_sense_exon_gene_df['strand'].map(strand_switch)
        merged_exon_df=pd.concat([exon_df, anti_sense_exon_gene_df]).reset_index(drop=True)
        
    else:
        merged_exon_df=df[(df.feature=='exon')|(df.feature=='gene')].copy()
        merged_exon_df['strand']='+-'
    
    if include_unannotated:
        non_coding_df=pd.read_csv(non_coding_path, sep="\t", header=None)
        non_coding_df.rename(columns={0:'chrom', 1:'start', 2:'end',3:'gene_name'}, inplace=True)
        non_coding_df['strand']="+-"
        non_coding_df['gene_type']="unannotated"
        non_coding_df['gene_id']=non_coding_df['gene_name']
        non_coding_df['exon_id']=non_coding_df['gene_id']    
        
        if check_strand:
            non_coding_df_plus=non_coding_df.copy()
            non_coding_df_plus['strand']="+"
            non_coding_df_minus=non_coding_df.copy()
            non_coding_df_minus['strand']="-"
            
            merged_exon_df=pd.concat([merged_exon_df, non_coding_df_plus, non_coding_df_minus]).reset_index(drop=True)
        else:
            merged_exon_df=pd.concat([merged_exon_df, non_coding_df]).reset_index(drop=True)
        
    merged_exon_df['strand_num']=merged_exon_df['strand'].map(strand_map)
    merged_exon_df['chrom_num']=merged_exon_df['chrom'].map(chrom_map)
    merged_exon_df['new_start']=1e10*merged_exon_df['chrom_num']+1e9*merged_exon_df['strand_num']+merged_exon_df['start']
    merged_exon_df['new_end']=1e10*merged_exon_df['chrom_num']+1e9*merged_exon_df['strand_num']+merged_exon_df['end']
    merged_exon_df=merged_exon_df.astype({'new_start': 'int', 'new_end': 'int'})

    exon_array=np.array(merged_exon_df)

    col_map={x:i for i,x in enumerate(merged_exon_df.columns)}

    gene_df=df[df.feature=='gene'].copy()
    gene_df['strand_num']=gene_df['strand'].map(strand_map)
    gene_df['chrom_num']=gene_df['chrom'].map(chrom_map)
    gene_df['new_start']=1e10*gene_df['chrom_num']+gene_df['start']
    gene_df['new_end']=1e10*gene_df['chrom_num']+gene_df['end']
    gene_df=gene_df.astype({'new_start': 'int', 'new_end': 'int'})
    gene_df.reset_index(drop=True, inplace=True)

    gene_ncls = NCLS64(gene_df.new_start.values, gene_df.new_end.values, gene_df.index.values)
    l_idx, r_idx=gene_ncls.all_overlaps_both(gene_df.new_start.values, gene_df.new_end.values, gene_df.index.values)
    overlapping_genes=set([tuple(sorted(x)) for x in zip(gene_df.iloc[l_idx].gene_id, gene_df.iloc[r_idx].gene_id) if x[0]!=x[1]])

    gene_id_to_name={x:y for x,y in zip(gene_df.gene_id, gene_df.gene_name)}
    gene_strand_map={x:y for x,y in zip(gene_df.gene_id, gene_df.strand)}
    gene_chrom_map={x:y for x,y in zip(gene_df.gene_id, gene_df.chrom)}
    
    trans_exon_counts=df[(df.feature=='exon')][['transcript_id']].groupby('transcript_id').value_counts()
    trans_exon_counts_map={x:trans_exon_counts.loc[x] for x in trans_exon_counts.index}
    
    exon_only_array=exon_array[exon_array[:,col_map['feature']]=='exon'][:,[col_map['gene_id'], col_map['transcript_id'], col_map['exon_cid']]]
    cluster_exon_tr_names=npi.group_by(list(zip(exon_only_array[:,0], exon_only_array[:,1])))
    split_exon_tr_names=cluster_exon_tr_names.split(np.arange(len(exon_only_array)))
    get_gene_exons={}
    for idx in split_exon_tr_names:
        chunk=exon_only_array[idx]
        gene, transcript=chunk[0,0], chunk[0,1]
        if gene not in get_gene_exons:
            get_gene_exons[gene]={}
        
        current_exons=list(chunk[:,2])
        current_exons_str=','.join(current_exons)
        get_gene_exons[gene][transcript]={'exons':current_exons, 'exons_string':current_exons_str}
                
    merged_exon_df=merged_exon_df.drop_duplicates('exon_cid')
    exon_array=np.array(merged_exon_df)
    
    return df, chrom_map, inv_chrom_map, merged_exon_df, exon_array, col_map, gene_df, gene_id_to_name, gene_strand_map, gene_chrom_map, overlapping_genes, trans_exon_counts_map, get_gene_exons