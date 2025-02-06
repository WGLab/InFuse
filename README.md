# InFuse
InFuse: gene fusion and exon-skipping detection from long read sequencing (under construction)
![overview](https://github.com/user-attachments/assets/25b97717-ad02-4125-896e-203dd038a7cc)

## Installation
```
git clone https://github.com/WGLab/InFuse.git
pip install numpy pandas numba sklearn pysam ncls numpy_indexed parasail
python InFuse/InFuse.py --help
```
## Usage

For gene fusion detection, please run the following command:

`python PATH_TO_INFUSE_REPO/InFuse.py --bam BAM_FILE --output OUTPUT_FOLDER --prefix PREFIX --gff GFF_ANNOTATION_FILE --transcripts TRANSCRIPT_FASTA_FILE --threads NUM_THREADS --min_support MIN_READS --distance_threshold DISTANCE_THRESHOLD --gf_only`

where BAM_FILE is an aligned BAM file, GFF_ANNOTATION_FILE is genome annotation GFF3 file, TRANSCRIPT_FASTA_FILE is a FASTA file containing spliced transcript sequences from the GFF3 file, NUM_THREADS is the number of threads to use for multiprocessing speedup, MIN_READS specifies the minimum read support needed for reporting a gene fusion, and DISTANCE_THRESHOLD specifies the distance in bp for merging gene fusion breakpoints.

The output folder will contain a tab seperated file `PREFIX.final_gf_double_bp` containing all gene fusions, as well as a python pickle file `PREFIX.final_gf_double_bp.pickle` containing detailed read-level information for all  gene fusions.
