## What is Paean 

Paean is designed for transcriptome quantification, especially for gene expression and alternative splicing profiling. Paean is written for CPU-GPU heterogeneous computing. So it accelerate the analysis procedure by 10000%

## How to use Paean

`paean gene_annotation your_bam ASE_file lib_for_sort`

gene_annotation: contain gene coordinates in gff3 format sorted according its position (the default is the gencode.sorted.gff3 )

your_bam: name sorted bam if in Paired-end. The reference should be identical with the gene and ASE file.

ASE_file:  ASE coordinates in gff3.  (the default is the SE.sorted.gff3 )

lib_for_sort: paean use two library for sort, thrust in CPU (this parameter is thrust )or in GPU(this parameter is cub)

### Example

`paean gencode.sorted.gff3 some.bam SE.sorted.gff3 cub`

## 








