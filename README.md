Kmer2vec-Pytorch
========
This implementation is based on the [original kmer2vec model by mais4719 (Magnus Isaksson)](https://github.com/mais4719/kmer2vec) as a Pytorch adaptation. 
The goal of this implementation is to make the code more optimized for performance, improve modularity, and 
to track runtime progress via tqdm.

In this implementation, I was unable to import pybedtools and thus resorted the use used of subprocess to call bedtools instead. 
This is required for kmer generation in the `reference_vocab_torch.py` script.

> #### Library requirements
* pytorch
* tqdm
* numpy
* pysam
> To run notebooks:
* biopython
* pandas
* scikit-learn
* scypy
* matplotlib
* six
* mpl_toolkits
> native libraries:
* subprocess
* os
* argparse
* sys
* multiprocessing
* collections
* glob
* threading<br>
> other tools:
* samtools - for indexing
* bedtools - for bed file processing as a replacement for pybedtools

### Pytorch script inputs
In the updated code, input arguments are passed via argparse. The parameters are largely the same as the original.

To run the reference vocabulary generation script:
```
python reference_vocab_torch.py \
    --fa reference_vocabulary/GRCh38_full_hs38d1_decoy.fa \
    --output reference_vocabulary/all_6-mers.tsv \
    --window_size 6 \ k-mer/window size in base pairs (default = 6 bp).
    -p 10 \ # number of threads to use
    # optional params
    --select 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16 \ #Select chromosomes to analyze.
    --filter_str "SINE,LINE" #Filter case-sensitive strings.
    --bed_file reference_vocabulary/GRCh38_full_hs38d1_decoy.fa.out.bed #Restrict the region(s) to analyze by one or more bed-files.
    
```

To run the kmer2vec pyrotch model:
```
python kmer2vec_torch.py \
    --fa reference_vocabulary/GRCh38_full_hs38d1_decoy.fa \
    --vocab reference_vocabulary/all_6-mers.tsv \
    --window_size 6 \ k-mer/window size in base pairs (default = 6 bp).
    --padding 10 \ # number of threads to use
    --embedding_size 100 \ # size of the embedding vector
    --num_neg_samples 100 \ # number of negative samples
    --learning_rate 0.8 \ # learning rate
    --epochs 10 \ # number of epochs
    --batch_size 1000 \ # batch size
    --output kmer2vec_embeddings.tsv # output file

```

<br>

---
From the [original README](https://github.com/mais4719/kmer2vec/blob/master/README.md):

kmer2vec
========
Unsupervised approach for feature (kmer embeddings) extraction from a provided reference genome. This
code is built on the word2vec model by Mikolov et al. You can find a good overview/tutorial within 
Tensorflow's tutorials [here](https://www.tensorflow.org/tutorials/word2vec).

#### Reference Vocabulary hg38
Folder ```reference_vocabulary``` contains code and a make file for downloading and pre-process the 
human reference genome GCA_000001405.15 GRCh38.

This will download the reference genome and repeatmasker, and create frequency of 6-mers in SINE, LINE, and ALL:
```
# make
```
Other k-mer sizes can be computed by running:
```
# make KMER_SIZE=8
```

#### Train on Reference
To start training on a reference genome (fasta file with corresponding faidx) run for example:
```
# ./kmer2vec.py \
    --fa_file reference_vocabulary/GRCh38_full_hs38d1_decoy.fa \
    --validation_vocabulary reference_vocabulary/all_6-mers.tsv \
    --min_kmer_size 3 \
    --max_kmer_size 8 \
    --padding 10 \
    --learning_rate 0.8 \
    --embedding_size 100 \
    --num_neg_samples 100
```

For debuging you can use an interactive session by adding the flagg ```--interactive```. This 
will make the program jump into an ipython shell after execution.

#### Notebooks
This is a work in progress. Notebooks are used to explore and find new ideas to improve the process.

##### Vector Cosine vs Nedleman-Wunsch Score
Looks at the correlation between cosine similarity and Nedleman-Wunsch score.

##### visualize_kmer_word2vec
Notebook visualize embeddings using t-SNEs and pre-computed SINE and LINE k-mer 
frequencies (created by make in folder "reference_vocabulary")

