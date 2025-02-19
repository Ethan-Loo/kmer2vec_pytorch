#!/usr/bin/env python3
"""
Program for calulating k-mer frequence within a fasta file.
"""
import os
import sys
import pysam
import click
import numpy as np
from collections import defaultdict
from collections import namedtuple
from pybedtools import BedTool
from multiprocessing.pool import Pool
from threading import Lock
from utils import patten2number
from utils import number2patten
from utils import kmer_vector2tsv_file
from utils import read_faidx

__author__ = 'Magnus Isaksson'
__credits__ = ['Magnus Isaksson']
__version__ = '0.1.0'

CHECK = '\N{CHECK MARK}'
BLUE = '\033[94m'
ENDC = '\033[0m'

# "Picklable" namedtuples
Job = namedtuple('Job', 'fasta_file chrom window_size regions')
Region = namedtuple('Region', 'start end')

lock = Lock()
final_result = None


def read_bed_regions(bed_files, chroms):
    """ Creates a merge region overall provided bed-files.

    Args:
        bed_files (list): Of bed-file paths (str)
           chroms (dict): With parents/chromosome id as key
                          and size (bp) as value.

    Returns:
        dict: A dictionary with chromosome/parent id as key
              and list of Region(start, end) (named tuple)
              object as value.
    """
    bed_file = BedTool(bed_files[0])
    if len(bed_files) > 1:
        bed_file = bed_file.cat(*bed_files[1:], postmerge=False)
    bed_file = bed_file.sort().merge()

    regions = defaultdict(list)
    for region in bed_file:
        if region.chrom in chroms.keys():
            regions[region.chrom].append(Region(region.start, region.end))
    return regions


def chrom_walker(pysam_ref, chrom, window_size):
    """ Kmer generator from a chromosome.

    Args:
          pysam_ref (FastaFile): pysam.FastaFile object.
                    chrom (str): Chromosome id.
              window_size (int): Windows/kmer size in bp.

    Returns:
        str: Sequence string generator.
    """
    chr_seq = pysam_ref.fetch(chrom)
    for pos in range(0, len(chr_seq) - window_size + 1):
        yield chr_seq[pos:pos + window_size]


def region_walker(pysam_ref, chrom, window_size, regions):
    """ Kmer generator from a chromosome sub-region.

    Args:
          pysam_ref (FastaFile): pysam.FastaFile object.
                    chrom (str): Chromosome id.
              window_size (int): Windows/kmer size in bp.
                 regions (list): List with regions (named tuple).

    Returns:
        str: Sequence string generator.
    """
    chr_seq = pysam_ref.fetch(chrom)
    for region in regions:
        kmers = (region.end - region.start) - window_size + 1
        for pos in range(region.start, region.start + kmers):
            yield chr_seq[pos:pos + window_size]


def kmer_counter(fasta_file, chrom, window_size, regions):
    """ Counts k-mers within a chromosome.

    Args:
              fasta_file (str): Path to indexed fasta file.
                   chrom (str): Chromosome id.
             window_size (int): Windows/kmer size in bp.
                       regions:
                           (list): Regions (named tuple).
                           (None): Ignore regions.

    Returns:
        tuple:
                  int: k-mer/windows size in bp.
                  str: Parent/Chromosome name
          numpy array: position = interger representation of k-mer sequnce
                       value = frequence within the chromosome(s)
    """
    kmer_count = np.array([0] * (4**window_size), dtype=np.uint32)

    with pysam.Fastafile(fasta_file) as ref:
        if regions:
            walker = region_walker(ref, chrom, window_size, regions)
        else:
            walker = chrom_walker(ref, chrom, window_size)

        for kmer_seq in walker:
            try:
                kmer_count[patten2number(kmer_seq.upper())] += 1
            except ValueError:
                # Simply ignore sequences contaning anything else
                # than A, C, G, T.
                #with open('debug.txt', 'aw') as f:
                #    f.write(kmer_seq + '\n')
                pass
    return (window_size, chrom, kmer_count)


def update_final_results(result):
    """ Callback function for multiprocessing pool.

    Args:
        result (tuple): 0: (int) window size
                        1: String or list of strings
                        2: numpy array
    """
    global final_result
    window_size, chrom, kmer_count = result

    if type(chrom) is list:
        chrom = ','.join(chrom)

    with lock:
        final_result += kmer_count
        result_str = '{seq}, Most Frequent for {chrom}'
        seq_winner = number2patten(kmer_count.argmax(), window_size)
        print(' ' + BLUE + CHECK + ENDC + ' ' +
              result_str.format(seq=seq_winner, chrom=chrom), flush=True)


@click.command()
@click.option('--output', '-o', type=click.Path(),
              help='Output tsv file.',
              required=True)
@click.option('--window_size', '-w', default=6, type=int,
              help='k-mer/windows size in base pairs (default = 6 bp).',
              required=False)
@click.option('--select', '-s', multiple=True,
              help='Specific selection of parent/chromosome ids. ' +
                   'For example: "-s chrX -s chrY" will only use the ' +
                   'sex chromosomes.',
              required=False)
@click.option('--filter_str', '-f', multiple=True,
              default=[],
              help='Filter case sensitive string(s). ' +
                   'For example: "-m decoy -m HLA" will filter all ' +
                   'parent/chromosom id contaning "decoy" or "HLA". ' +
                   '(Filter examples "decoy", "chrEBV", "HLA", "alt")',
              required=False)
@click.option('--bed_file', '-b', type=click.Path(exists=True), multiple=True,
              help='Restrict the region(s) to analys by one, or more, ' +
                   'bed-file(s).',
              required=False)
@click.option('--processes', '-p', default=1, type=int,
              help='Number of parallel processes allowed (default = 1).',
              required=False)
@click.argument('fasta_file', type=click.Path(exists=True))
def main(fasta_file, output, window_size=6, select=None,
         filter_str=[], bed_file=None, processes=1):

    global final_result

    # Try to find fai file.
    fai_candidates = [fasta_file + '.fai',
                      fasta_file.rstrip('.fa') + '.fai',
                      fasta_file.rstrip('.gz') + '.fai']

    try:
        faidx_file = [os.path.isfile(c) for c in fai_candidates].index(True)
        faidx_file = fai_candidates[faidx_file]
    except ValueError:
        print('Not able to find a index file (*.fai) for %s' % fasta_file)
        return

    # Load parent/chromosome id from index file.
    chroms = read_faidx(faidx_file,
                        filter_strs=filter_str)

    # Subselect list of parent/chromosome id if needed.
    if select:
        try:
            chroms = {s: chroms[s] for s in select}
        except KeyError as e:
            print('Parent {} is not be found in {}.'.format(e, faidx_file))
            return

    # Any bed file(s)?
    bed_dict = {}
    if bed_file:
        print('\n ' + BLUE + CHECK + ENDC + ' ' +
              'Parsing and merging regions from provided bed-file(s).',
              flush=True)
        bed_dict = read_bed_regions(bed_file, chroms)

    # Create parallel jobs for pool and run.
    print('')
    jobs = [Job(fasta_file, chrom, window_size, None)
            for chrom in chroms.keys()]

    final_result = np.array([0] * (4**window_size), dtype=np.uint32)
    pool = Pool(processes=processes)
    runs = []
    for job in jobs:

        if bed_file:
            # Create a new Job object with updated regions.
            # Add empty list if no regions in bed-file(s).
            job = Job(*job[:-1], regions=bed_dict.get(job.chrom, []))

        run = pool.apply_async(kmer_counter,
                               job,
                               callback=update_final_results)
        runs.append(run)

    for run in runs:
        run.wait()
        if not run.successful():
            print(run.get())

    # Save results to tsv text file.
    kmer_vector2tsv_file(output,
                         final_result,
                         min_length=window_size,
                         max_length=window_size)
    print('\nDone...\n')


if __name__ == '__main__':

    main()

