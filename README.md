# MACIE (Multi-dimensional Annotation Class Integrative Estimation)

## Description
Thank you for your interest in MACIE. MACIE (Multi-dimensional Annotation Class Integrative Estimation) is an unsupervised multivariate mixed model framework to assess multi-dimensional functional impacts for both coding and non-coding variants in the human genome. MACIE integrates a variety of functional annotations, including protein function scores, evolutionary conservation scores, and epigenetic annotations from ENCODE and Roadmap Epigenomics, and estimates the joint posterior probabilities of each genetic variant being functional.


## Data Availability and Code Reproducibility
The MACIE scores (and other integrative scores) used in all benchmarking examples are available for download [here](https://drive.google.com/drive/folders/1gzqsfgaO1WCh5pAQUgVlUNsX9HYneO7p?usp=sharing). Precomputed MACIE scores for every possible variant in the human genome are available for download via Zenodo: [Part 1 (Chr1 - Chr3)](https://zenodo.org/record/5755656), [Part 2 (Chr4 - Chr7)](https://zenodo.org/record/5756449), [Part 3 (Chr8 - Chr13)](https://zenodo.org/record/5756479), [Part 4 (Chr14 - Chr22)](https://zenodo.org/record/5756563). These are compressed with the bgzip utility, and indexed with tabix, both of which are part of the [Samtools software suite](https://www.htslib.org). In addition, tabix provides a means of efficiently extracting subsets of the data defined by genomic regions. For example, the command line

`tabix MACIE_hg19_noncoding_chr1.tab.bgz 1:20000-30000 > Subset.txt`

extracts all variants on chromosome 1 from position 20,000 through 30,000 and writes them to the file Subset.txt. In this example, the tabix index file, `MACIE_hg19_noncoding_chr1.tab.bgz.tbi`, needs to be in the same directory as the main data file, `MACIE_hg19_noncoding_chr1.tab.bgz`. Samtools, including bgzip and tabix, is available [here](https://www.htslib.org/download). 

The code used for training MACIE models are available [here](https://github.com/xihaoli/MACIE/blob/main/code/MACIE.py).

All genomic coordinates are given in NCBI Build 37/UCSC hg19.

## Reference
Xihao Li*, Godwin Yung*, Hufeng Zhou, Ryan Sun, Zilin Li, Kangcheng Hou, Martin Jinye Zhang, Yaowu Liu, Theodore Arapoglou, Chen Wang, Iuliana Ionita-Laza, and Xihong Lin (2021+) "A Multi-Dimensional Integrative Scoring Framework for Predicting Functional Variants in the Human Genome". *Submitted*.
