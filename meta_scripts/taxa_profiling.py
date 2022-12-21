import argparse
import sys
import os
import math
from collections import defaultdict
import multiprocessing as mp

def load_reads(args):
    # with gzip.open(args.fastq, 'rt') as handle:
    with open(args.fastq, 'r') as handle:
        content = handle.readlines()
        records = [''.join(content[i:i+4]) for i in range(0, len(content), 4)]
        # args.reads = {rec.split('\n')[0]: rec for rec in records}
        print(len(records))

def parse_data(data, args, results, process_id):
    process_results = defaultdict(int)
    for line in data:
        taxon = args.dl_toda_taxonomy[int(line.split('\t')[2])].split('\t')[args.ranks[args.rank]]
        prob_score = float(line.split('\t')[3])
        label = int(line.split('\t')[2])
        if prob_score > args.cutoff:
            process_results[label] += 1
            if args.binning:
                fq_filename = os.path.join(args.output_dir, f'{label}-{process_id}.fq') if args.rank == 'species' else os.path.join(args.output_dir, f'{taxon}-{process_id}.fq')
                with open(fq_filename, 'a') as out_fq:
                    out_fq.write(args.reads[read_id])

    results[process_id]= process_results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dl_toda_output', type=str, help='output file with predicted species obtained from running DL-TODA', required=True)
    parser.add_argument('--fastq', type=str, help='path to sample gzipped fastq file', required=('--binning' in sys.argv))
    parser.add_argument('--binning', help='bin reads', action='store_true')
    parser.add_argument('--processes', type=int, help='number of processes', default=mp.cpu_count())
    parser.add_argument('--output_dir', type=str, help='path to output directory', default=os.getcwd())
    parser.add_argument('--rank', type=str, help='taxonomic rank at which the analysis should be done', default='species')
    parser.add_argument('--cutoff', type=float, help='cutoff or probability score between 0 and 1 above which reads should be analyzed', default=0.0)
    parser.add_argument('--taxa', nargs='+', default=[], help='list of taxa to bin')
    parser.add_argument('--tax_db', help='type of taxonomy database used in DL-TODA', choices=['ncbi', 'gtdb'], required=True)
    args = parser.parse_args()

    args.ranks = {'phylum': 5, 'class': 4, 'order': 3, 'family': 2, 'genus': 1, 'species': 0}

    # get dl-toda taxonomy
    if args.tax_db == 'ncbi':
        index = 2
    elif args.tax_db =='gtdb':
        index = 1

    # update and create output directory
    args.output_dir = os.path.join(args.output_dir, '-'.join(args.dl_toda_output.split('/')[-1].split('-')[:-1]), f'cutoff-{args.cutoff}')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # define name of output file with taxonomic profiles
    args.output_file = os.path.join(args.output_dir, '-'.join(args.dl_toda_output.split('/')[-1].split('-')[:-1]) + f'-cutoff-{args.cutoff}-out.tsv')

    if args.binning:
        # load reads
        load_reads(args)

    args.dl_toda_taxonomy = {}
    with open('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]) + '/data/dl_toda_taxonomy.tsv', 'r') as in_f:
        for line in in_f:
            line = line.rstrip().split('\t')
            args.dl_toda_taxonomy[int(line[0])] = '\t'.join(line[index].split(';'))

    with open(args.dl_toda_output, 'r') as f:
        content = f.readlines()
    chunk_size = math.ceil(len(content)/args.processes)
    data_split = [content[i:i+chunk_size] for i in range(0,len(content),chunk_size)]
    print(chunk_size, args.processes, len(data_split), len(data_split[0]))

    with mp.Manager() as manager:
        results = manager.dict() # dictionary with key = read id and value = predicted taxon
        processes = [mp.Process(target=parse_data, args=(data_split[i], args, results, i)) for i in range(len(data_split))]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        print(f'#reads: {len(results)}')

        # create file with taxonomic profiles
        with open(args.output_file, 'w') as out_f:
            for k, v in args.dl_toda_taxonomy.items():
                num_reads = 0
                for process_id, process_results in results.items():
                    for label, read_count in process_results.items():
                        if label == k:
                            num_reads += read_count
                out_f.write(f'{v}\t{num_reads}\n')
