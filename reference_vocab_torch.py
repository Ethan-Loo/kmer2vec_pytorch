#!/usr/bin/env python3
"""
Program for calculating k-mer frequency within a fasta file.
"""
import os
import sys
import pysam
import argparse
import numpy as np
import subprocess
from collections import defaultdict
from collections import namedtuple
# from pybedtools import BedTool
from multiprocessing.pool import Pool
from threading import Lock
from utils_torch import patten2number
from utils_torch import number2patten
from utils_torch import kmer_vector2tsv_file
from utils_torch import read_faidx

__author__ = 'Magnus Isaksson', 'modifications by Ethan Loo'
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
    """ Creates a merged region overall provided bed-files using bedtools.

    Args:
        bed_files (list): List of bed-file paths (str)
        chroms (dict): Dictionary with parent/chromosome id as key
                       and size (bp) as value.

    Returns:
        dict: A dictionary with chromosome/parent id as key
              and list of Region(start, end) (named tuple)
              object as value.
    """
    merged_bed_file = "merged.bed"

    # Use bedtools to merge the bed files
    bedtools_cmd = ["bedtools", "merge", "-i"] + bed_files + ["-c", "1,2,3", "-o", "distinct"]

    try:
        subprocess.run(bedtools_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running bedtools: {e}")
        return {}

    # Read the merged bed file and create regions
    regions = defaultdict(list)
    with open(merged_bed_file, "r") as merged_bed:
        regions = {
            chrom: [
                Region(int(fields[1]), int(fields[2]))
                for fields in (line.strip().split("\t"))
                if chrom == fields[0]
            ]
            for line in merged_bed
            if line.strip().split("\t")[0] in chroms.keys()
        }

    # Clean up the temporary merged bed file
    subprocess.run(["rm", merged_bed_file])

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
    # for pos in range(0, len(chr_seq) - window_size + 1):
    #     yield chr_seq[pos:pos + window_size]

    return (chr_seq[pos:pos + window_size] for pos in range(0, len(chr_seq) - window_size + 1))


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
    # for region in regions:
    #     kmers = (region.end - region.start) - window_size + 1
    #     for pos in range(region.start, region.start + kmers):
    #         yield chr_seq[pos:pos + window_size]

    return (
        chr_seq[pos:pos + window_size]
        for region in regions
        for pos in range(region.start, region.start + (region.end - region.start) - window_size + 1)
    )

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
          numpy array: position = integer representation of k-mer sequence
                       value = frequency within the chromosome(s)
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
                # Simply ignore sequences containing anything else
                # than A, C, G, T.
                # with open('debug.txt', 'aw') as f:
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


def main():
    global final_result

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Calculate k-mer frequency within a fasta file.")

    # Define command-line arguments
    parser.add_argument('--fasta_file', '-fa',type=str, help='Path to indexed fasta file.')
    parser.add_argument('--output', '-o', type=str, help='Output tsv file.', required=True)
    parser.add_argument('--window_size', '-w', default=6, type=int, help='k-mer/window size in base pairs (default = 6 bp).')
    parser.add_argument('--select', '-s', action='append', help='Specific selection of parent/chromosome ids.')
    parser.add_argument('--filter_str', '-f', action='append', default=[], help='Filter case-sensitive strings.')
    parser.add_argument('--bed_file', '-b', type=str, nargs='+', help='Restrict the region(s) to analyze by one or more bed-files.')
    parser.add_argument('--processes', '-p', default=1, type=int, help='Number of parallel processes allowed (default = 1).')

    # Parse command-line arguments
    args = parser.parse_args()

    # Try to find fai file.
    fai_candidates = [args.fasta_file + '.fai',
                      args.fasta_file.rstrip('.fa') + '.fai',
                      args.fasta_file.rstrip('.gz') + '.fai']

    try:
        faidx_file = [os.path.isfile(c) for c in fai_candidates].index(True)
        faidx_file = fai_candidates[faidx_file]
    except ValueError:
        print('Not able to find an index file (*.fai) for %s' % args.fasta_file)
        return

    # Load parent/chromosome id from index file.
    chroms = read_faidx(faidx_file, filter_strs=args.filter_str)

    # Subselect list of parent/chromosome id if needed.
    if args.select:
        try:
            chroms = {s: chroms[s] for s in args.select}
        except KeyError as e:
            print('Parent {} is not found in {}.'.format(e, faidx_file))
            return

    # Any bed file(s)?
    bed_dict = {}
    if args.bed_file:
        print('\n ' + BLUE + CHECK + ENDC + ' ' +
              'Parsing and merging regions from provided bed-file(s).',
              flush=True)
        bed_dict = read_bed_regions(args.bed_file, chroms)

    # Create parallel jobs for pool and run.
    print('')
    jobs = [Job(args.fasta_file, chrom, args.window_size, None)
            for chrom in chroms.keys()]

    final_result = np.array([0] * (4 ** args.window_size), dtype=np.uint32)
    pool = Pool(processes=args.processes)

    runs = [
        pool.apply_async(kmer_counter,
                         Job(*job[:-1], regions=bed_dict.get(job.chrom, [])),
                         callback=update_final_results)
        for job in jobs
    ]


    for run in runs:
        run.wait()
        if not run.successful():
            print(run.get())

    # Save results to a TSV text file.
    kmer_vector2tsv_file(args.output,
                         final_result,
                         min_length=args.window_size,
                         max_length=args.window_size)
    print('\nDone...\n')

if __name__ == '__main__':
    main()
