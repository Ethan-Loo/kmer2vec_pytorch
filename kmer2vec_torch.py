#!/usr/bin/env python3
"""
Vector Representations of genomic k-mers
(Skip-gram Model)
Original Author: Magnus Isaksson
Converted to Pytorch with modifications by: Ethan Loo
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from random import shuffle
import argparse
import pysam
from tempfile import mkdtemp
from tqdm import tqdm
from utils_torch import multisize_patten2number, number2multisize_patten
from utils_torch import tsv_file2kmer_vector
from utils_torch import read_faidx
from utils_torch import reverse_complement, gc, sequence_entropy


class Kmer2Vec(nn.Module):
    """ Kmer2Vec model."""
    def __init__(self, args):
        super(Kmer2Vec, self).__init__()
        self.args = args
        self.save_path = args.save_path
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # Set the device here
        self.embedding_size = args.embedding_size
        assert args.min_kmer_size <= args.max_kmer_size
        self.kmer_sizes = torch.arange(args.min_kmer_size, args.max_kmer_size + 1)

        self.vocab_size = torch.sum(4 ** self.kmer_sizes)

        self.VALID_WINDOW = 200
        self.VALID_SET_SIZE = 10
        self.valid_kmers = self._get_valid_examples()

        self.ALLOWED_CHRS = [chr_name for chr_name in
                             read_faidx(args.fa_file, args.chr_filter.split(','))]

        self.NUMBER_OF_CHRS = len(self.ALLOWED_CHRS)

        self.build_graph()

        # self.metadata_file = create_metadata(os.path.join(args.save_path,
        #                                     f'{self.save_path}metadata{args.min_kmer_size}{args.max_kmer_size}.tsv'),
        #                                      args.min_kmer_size,
        #                                      args.max_kmer_size)
        self.metadata_filename = os.path.join(args.save_path,
                                              f'{self.save_path}metadata{args.min_kmer_size}_{args.max_kmer_size}.tsv')


    def setup_summaries(self):
        """ Setting up logging and summary output. """
        args = self.args

        # Initialize SummaryWriter
        writer = SummaryWriter(os.path.join(args.save_path, 'train_logs'))

        # Summary variables
        loss_op = self.loss.item()  # Assuming self.loss is a tensor representing the loss
        nce_weights_op = self.nce_weights  # Assuming self.nce_weights is a tensor
        nce_biases_op = self.nce_biases  # Assuming self.nce_biases is a tensor
        embeddings_op = self.embeddings.weight  # Assuming self.embeddings is an embedding layer
        norm_embeddings_op = torch.norm(self.embeddings.weight, dim=1)  # Normalized embeddings

        # Writing the summaries
        writer.add_scalar('Batch average NCE loss', loss_op, self.global_step)
        writer.add_histogram('NCE weights', nce_weights_op, self.global_step)
        writer.add_histogram('NCE biases', nce_biases_op, self.global_step)
        writer.add_histogram('Embeddings', embeddings_op, self.global_step)
        writer.add_histogram('Normalized Embeddings', norm_embeddings_op, self.global_step)

        # Embedding visualization
        # Note: PyTorch doesn't have built-in functionality for embedding visualization like TensorFlow's TensorBoard projector.
        # You might need to export the embeddings and metadata manually for visualization using other tools.

        # Close the SummaryWriter when done
        writer.close()

    def _get_valid_examples(self):
        """ Get random examples from validation vocabulary file."""
        valid_file = self.args.validation_vocabulary

        try:
            kmer_vector = tsv_file2kmer_vector(valid_file,
                                               self.args.min_kmer_size,
                                               self.args.max_kmer_size)

            valid_kmers = np.argpartition(kmer_vector, -self.VALID_WINDOW)[-self.VALID_WINDOW:]
            valid_kmers = np.random.choice(valid_kmers,
                                           self.VALID_SET_SIZE,
                                           replace=False)
        except Exception:
            print('Not able to fetch validation examples ' +
                  'from validation vocabulary file ' +
                  '[{}].'.format(valid_file))
            raise

        return valid_kmers

    def build_graph(self):
        """ Build the computational graph."""
        args = self.args

        self.train_inputs = torch.LongTensor(args.batch_size).to(self.device)
        self.train_labels = torch.LongTensor(args.batch_size, 1).to(self.device)
        self.valid_dataset = torch.LongTensor(self.valid_kmers)

        self.embeddings = nn.Embedding(self.vocab_size, args.embedding_size).to(self.device)

        # Initialize embeddings if provided
        if args.embeddings_file:
            print('Loading embeddings from [%s]' % args.embeddings_file)
            self.embeddings.weight.data = torch.from_numpy(np.load(args.embeddings_file))

        self.nce_weights = nn.Parameter(torch.randn(self.vocab_size, args.embedding_size)).to(self.device)
        self.nce_biases = nn.Parameter(torch.zeros(self.vocab_size)).to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()

        self.global_step = 0

    def forward(self, target, context):
        target_embed = self.embeddings(target)
        context_embed = self.embeddings(context)

        # Expand the dimensions of context_embed to match the shape of self.nce_weights
        context_embed = context_embed.view(-1, self.args.embedding_size)

        # Calculate the scores as in the original code
        scores = torch.matmul(context_embed, self.nce_weights.t()) + self.nce_biases

        # Compute the loss
        loss = self.loss_fn(scores, target.view(-1))

        # Store the loss as an attribute
        self.loss = loss

        return scores, target_embed

    def calculate_cosine_similarity(self):
        with torch.no_grad():
            normalized_embeddings = self.embeddings.weight / torch.norm(self.embeddings.weight, dim=1, keepdim=True)
            valid_embeddings = normalized_embeddings[self.valid_dataset]
            similarity = torch.matmul(valid_embeddings, normalized_embeddings.t())
        return similarity

    def save_vocab(self, filename):
        try:
            np.save(filename, self.embeddings.weight.data.cpu().numpy())
        except Exception as e:
            print('Unable to save data to', filename, ':', e)

    def train(self):
        args = self.args
        optimizer = optim.RAdam(self.parameters(), lr=args.learning_rate)
        writer = SummaryWriter(os.path.join(args.save_path, 'train'))

        embedding_saver = torch.save
        output_file_template = os.path.join(args.save_path,
                                            'max{max_size}_min{min_size}_mers_{padding}padding_{emb_size}embedding-size_epoch{epoch}_batch{index}')

        for epoch in range(1, args.epochs + 1):
            old_chrom = None
            chroms_done = 0
            shuffle(self.ALLOWED_CHRS)

            # for index, (chrom, batch, labels) in enumerate(batch_generator(args.batch_size,
            #                                                                args.fa_file,
            #                                                                self.ALLOWED_CHRS,
            #                                                                args.min_kmer_size,
            #                                                                args.max_kmer_size,
            #                                                                args.padding)):
            total_batches = calculate_total_batches(args.fa_file,
                                                    self.ALLOWED_CHRS,
                                                    args.batch_size,
                                                    args.min_kmer_size,
                                                    args.max_kmer_size,
                                                    args.padding)

            for index, (chrom, batch, labels) in tqdm(enumerate(batch_generator(args.batch_size,
                                                                                args.fa_file,
                                                                                self.ALLOWED_CHRS,
                                                                                args.min_kmer_size,
                                                                                args.max_kmer_size,
                                                                                args.padding)),
                                                      total=total_batches,
                                                      desc='Batches'):

                if old_chrom != chrom:
                    print('Starting training on {}...'.format(chrom))
                    print(f'Epoch {epoch} ({chroms_done}/{self.NUMBER_OF_CHRS} done)')
                    chroms_done += 1
                    old_chrom = chrom

                self.train_inputs.data.copy_(torch.LongTensor(batch)).to(self.device)
                self.train_labels.data.copy_(torch.LongTensor(labels)).to(self.device)

                optimizer.zero_grad()
                scores, _ = self(self.train_inputs, self.train_labels)
                loss = self.loss_fn(scores, self.train_labels.view(-1))
                loss.backward()
                optimizer.step()

                writer.add_scalar('Batch average NCE loss', loss.item(), self.global_step)

                info_str = '{} ({}/{} done) epoch: {} batch: {} loss: {}'
                # if the index is a multiple of 100, print the info string
                # if index % 100 == 0:
                #     print(info_str.format(chrom,
                #                           chroms_done,
                #                           self.NUMBER_OF_CHRS,
                #                           epoch,
                #                           index))

                self.global_step += 1

                if index % 20000 == 0:
                    print(info_str.format(chrom,
                                          chroms_done,
                                          self.NUMBER_OF_CHRS,
                                          epoch,
                                          index,
                                          loss.item()))

                if index % 50000 == 0:
                    sim = self.calculate_cosine_similarity()
                    for i in range(self.VALID_SET_SIZE):
                        valid_kmer = number2multisize_patten(self.valid_kmers[i],
                                                             args.min_kmer_size,
                                                             args.max_kmer_size,
                                                             self.device)
                        top_k = 4  # Number of nearest neighbors
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        close_kmers = [number2multisize_patten(nearest[k],
                                                               args.min_kmer_size,
                                                               args.max_kmer_size,
                                                               self.device)
                                       for k in range(top_k)]
                        print('Nearest to {} -> ({})'.format(valid_kmer, ', '.join(close_kmers)))

                    embedding_saver(self.state_dict(), os.path.join(args.save_path, 'model.pth'))

                if index % 500000 == 0:
                    self.save_vocab(output_file_template.format(min_size=args.min_kmer_size,
                                                                max_size=args.max_kmer_size,
                                                                padding=args.padding,
                                                                emb_size=args.embedding_size,
                                                                epoch=epoch,
                                                                index=index))

            self.save_vocab(output_file_template.format(min_size=args.min_kmer_size,
                                                        max_size=args.max_kmer_size,
                                                        padding=args.padding,
                                                        emb_size=args.embedding_size,
                                                        epoch=epoch,
                                                        index=index))
        print(info_str.format(chrom,
                              chroms_done,
                              self.NUMBER_OF_CHRS,
                              epoch,
                              index,
                              loss.item()))

        # After training, create the metadata file with embeddings
        metadata_filename = os.path.join(f'{self.save_path}metadata{self.args.min_kmer_size}_{self.args.max_kmer_size}.tsv')
        create_metadata(metadata_filename, self, self.args.min_kmer_size, self.args.max_kmer_size)

        # Update the self.metadata_filename to point to the newly created metadata file
        self.metadata_filename = metadata_filename

        self.setup_summaries()

def calculate_total_batches(fa_file, chroms, batch_size, min_length, max_length, padding):

    total_batches = 0

    for chrom in chroms:
        with pysam.FastaFile(fa_file) as ref:
            chr_seq = ref.fetch(chrom)
            num_possible_batches = len(chr_seq) - 2 * padding - 2 * max_length + 1
            num_batches = math.ceil(num_possible_batches / batch_size) * 2 # Account for reverse complement
            total_batches += num_batches

    return total_batches


def context_generator(fa_file, chroms, min_length=3, max_length=5, padding=1):
    """ Creates context and target k-mers using provided fasta and
    fasta index file. Using a 1 base sliding window approach with
    random k-mer sizes between min and max length. Both polarities
    are sampled randomly.

    E.g. min_length=3, max_length=5, padding=1

        rnd_kmer_sizes = [4, 3, 5]
        CATATCA -> ['CATA', 'ATA', 'TATCA']

        -> ('chr?', 'ATA', ['CATA', 'TATCA'])

        DNA sequences will be converted into ints for the final result

        -> ('chr?', 12, [140, 1140])

    Args:
          fa_file (str): Path to fasta file with with accompanying
                         Samtools index file (*.fai).
          chroms (list): Orded list of chromosome/parent ids which will
                         be included when iterating over the fasta file.
       min_length (int): Minimal allowed kmer size (nt).
       max_length (int): Maximum allowed kmer size (nt).
          padding (int): Number of kmers, on each side, added to the context.

    Yields:
        chromosom_id (str), target_seq (int), list(context_seqs (ints))
    """
    kmer_sizes = np.arange(min_length, max_length + 1)
    print('Using kmer sizes:', kmer_sizes)

    with pysam.FastaFile(fa_file) as ref:
        for chrom in chroms:
            chr_seq = ref.fetch(chrom)
            for subseq_pos in range(0, len(chr_seq)):
                rnd_kmer_sizes = np.random.choice(kmer_sizes, padding * 2 + 1)
                subseq = chr_seq[subseq_pos:subseq_pos +
                                            rnd_kmer_sizes.size + rnd_kmer_sizes.max()].upper()

                if len(subseq) < rnd_kmer_sizes.size + rnd_kmer_sizes.max():
                    continue

                if np.random.randint(2):
                    subseq = reverse_complement(subseq)

                try:
                    num_kmers = []

                    for i, pos in enumerate(rnd_kmer_sizes):
                        kmer_seq = subseq[i:i + rnd_kmer_sizes[i]]
                        number_seq = multisize_patten2number(kmer_seq,
                                                             min_length,
                                                             max_length)
                        num_kmers.append(number_seq)

                    context = np.array(num_kmers[:padding] +
                                       num_kmers[-padding:])
                    target = num_kmers[padding]

                    i += 1

                    yield chrom, target, context

                except (KeyError, IndexError, ValueError):
                    pass

def batch_generator(batch_size, fa_file, chroms,
                    min_length=3, max_length=5, padding=1):
    """ Target and context k-mer batch generator for a reference fasta file.

    Args:
        batch_size (int): Size of each yield batch.
           fa_file (str): Path to fasta file with with accompanying
                          Samtools index file (*.fai).
           chroms (list): Orded list of chromosome/parent ids which will
                          be included when iterating over the fasta file.
        min_length (int): Minimal allowed kmer size (nt).
        max_length (int): Maximum allowed kmer size (nt).
           padding (int): Number of kmers, on each side, added to the context.

    Yields:
        chromosom_id (str),
        y_batch (numpy uint32 array shape=(batch_size)),
        label_batch (numpy uint32 array shape=(batch_size, 1))

    """
    y_batch = torch.zeros(batch_size, dtype=torch.long)
    label_batch = torch.zeros(batch_size, 1, dtype=torch.long)
    i = 0

    for chrom, target, context in context_generator(fa_file, chroms,
                                                    min_length, max_length,
                                                    padding):
        for neighbour in context:
            y_batch[i] = target
            label_batch[i] = neighbour
            if i % (batch_size - 1) == 0 and i > 0:
                yield chrom, y_batch, label_batch
                y_batch = torch.zeros(batch_size, dtype=torch.long)
                label_batch = torch.zeros(batch_size, 1, dtype=torch.long)
                i = 0
            else:
                i += 1

def create_metadata(filename, model, min_length=3, max_length=5):
    print('Creating metadata file:', filename)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kmer_sizes = np.arange(min_length, max_length + 1)
    vocabulary_size = np.sum(4 ** kmer_sizes)

    # Determine the embedding size
    embedding_size = model.embedding_size

    # Generate the column names dynamically
    column_names = ['name', 'length', 'gc', 'entropy']
    embedding_columns = [f'embedding_{i}' for i in range(embedding_size)]
    norm_embedding_columns = [f'norm_embedding_{i}' for i in range(embedding_size)]
    column_names.extend(embedding_columns)
    column_names.extend(norm_embedding_columns)

    # Create placeholders for each column with the correct number of placeholders
    placeholder = '\t'.join(['{}'] * len(column_names))

    # Use the generated column names in tmpl_str
    tmpl_str = '\t'.join(column_names) + '\n'

    # Generate all possible sequences
    seq_numbers = np.arange(0, vocabulary_size)
    seqs = [number2multisize_patten(seq_numb, min_length, max_length, device) for seq_numb in tqdm(seq_numbers)]

    # Calculate GC and sequence entropy for all sequences
    gc_values = [round(gc(seq), 2) for seq in seqs]
    entropy_values = [round(sequence_entropy(seq), 2) for seq in seqs]

    # Calculate embeddings for all sequences
    embeddings = model.embeddings(torch.LongTensor(seq_numbers).to(device))
    # Calculate the normalized embeddings for all sequences
    norm_embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)

    with open(filename, 'w') as f:
        # Use the generated column names
        f.write(tmpl_str)

        for seq_numb, seq, gc_val, entropy_val, embedding, norm_embedding in zip(seq_numbers, seqs, gc_values, entropy_values, embeddings, norm_embeddings):
            # Create lists of embedding and norm_embedding values
            embedding_values = embedding.detach().cpu().numpy()
            norm_embedding_values = norm_embedding.detach().cpu().numpy()

            # Format embedding and norm_embedding values as strings
            embedding_str = '\t'.join(str(val) for val in embedding_values)
            norm_embedding_str = '\t'.join(str(val) for val in norm_embedding_values)

            # Write the row to the file
            f.write(placeholder.format(seq, len(seq), gc_val, entropy_val, *embedding_values, *norm_embedding_values) + '\n')

    return filename



def main(args):

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", torch.cuda.get_device_name(0))

    model = Kmer2Vec(args).to(device)
    model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vector Representations of genomic k-mers (Skip-gram Model)')
    parser.add_argument('--fa_file', type=str, default='', help='Fasta file with accompanied samtool index.')
    parser.add_argument('--chr_filter', type=str, default='decoy,chrEBV',
                        help='Comma-separated list of strings not allowed within the chromosome/parent name.')

    parser.add_argument('--validation_vocabulary', '-v', type=str, default='', help='Path to validation vocabulary file.')
    parser.add_argument('--save_path', type=str, default='.', help='Output path for logs and results.')

    parser.add_argument('--min_kmer_size', type=int, default=3, help='Minimum kmer size in nucleotides used during training.')
    parser.add_argument('--max_kmer_size', type=int, default=5, help='Maximum kmer size in nucleotides used during training.')
    parser.add_argument('--padding', type=int, default=1, help='Number of kmers used on each side as the context for this skip gram model.')

    parser.add_argument('--batch_size', type=int, default=128, help='Number of training examples processed per step.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='Initial learning rate.')
    parser.add_argument('--embedding_size', type=int, default=100, help='The embedding dimension size.')
    parser.add_argument('--num_neg_samples', type=int, default=100, help='Negative samples per training example.')

    parser.add_argument('--interactive', action='store_true', help='Jumps into an iPython shell for debugging.')
    parser.add_argument('--embeddings_file', type=str, default=None, help='Path to previously embeddings to load (numpy format).')

    args = parser.parse_args()
    main(args)
