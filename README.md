Kmer2vec-Pytorch
========
This implementation is based on the [original kmer2vec model by mais4719 (Magnus Isaksson)](https://github.com/mais4719/kmer2vec) as a Pytorch adaptation. 
The goal of this implementation is to make the code more optimized for performance, improve modularity, and 
track runtime progress via tqdm.

In this implementation, I was unable to import pybedtools and thus resorted the use used of subprocess to call bedtools instead. 
This is required for kmer generation in the `reference_vocab_torch.py` script.

> #### Library requirements
* pytorch (including cudatoolkit if using GPU)
* tqdm
* numpy
* pysam
* pybedtools (not used in this implementation, but needed for the original implementation)
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

#### Known issues

The `Batches` TQDM progress bar currently lacks accuracy during the training process.
The total amount of training batches is currently estimated by a rough approximation function that requires refinement.

Hanging during chromosome selection:
- The training process is stalled when only one chromosome is used in the fasta file. 
- On occasion, the training process will hang when multiple chromosomes are selected as well. This does not occur consistently and may be due to a memory leak or a deadlock in the code.

### Notes about the key changes made to this version

There are multiple changes made in this implementation:
1. The optimization algorithm is changed from SGD to RAdam. 
2. The original code did not sure the embeddings in the tsv file. This is added to the code. 
The embeddings may still need to be parsed from their respective columns to be used in downstream applications.
3. TQDM has been added to track the progress of the library generation in the metadata and to trask training progress. 
This allows for better tracking of the runtime and estimated completion time and allows for the user to see the progress of the code.
4. There are some minor loop optimizations into comprehensions improve runtime performance.

#### Notes on kmer scaling and training time

As the kmer size increases, the number of kmers increases exponentially. The training time scales with the number and size of the kmers due to their relational complexity.
The training time is also dependent on the number of negative samples and the embedding size. 
Larger kmers will require more resources to train including a higher CPU core count and GPU memory.

As an example, training a 5-mer model with a batch size of 512, 100 negative samples, and an embedding size of 5 on the entire GRCh38 
Homo Sapiens reference assembly will take approximately 56 hours/epoch. This level of embedding and kmer size is small 
enough for more modest resources: using up to 2 CPU cores. 5GB of RAM, and approx. 2.5 GB of VRAM. <br>
Training a 10-mer model required considerably higher resources at approximately 8 cores, 15GB VRAM, and the more GPU compute capacity.

Model convergence appears to occur quickly, however, suggesting that a full reference assembly may not be necessary to obtain accurate embeddings.
More time and testing are needed to confirm this. Currently, the metadata are not generated until training is complete and embeddings are only saved between epochs in a numpy file. 

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

