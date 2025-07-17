import argparse
import sys
import os
import math
import glob
import gzip
import json
import shutil
from collections import defaultdict
import multiprocessing as mp
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
from vis_scripts.parse_tool_output import *
from dataprep_scripts.ncbi_tax_utils import parse_nodes_file, parse_names_file

def LoadReads(args):
    # get reads from fastq file
    if args.fastq[-2:] == 'gz':
        with gzip.open(args.fastq, 'rt') as handle:
            content = handle.readlines()
    else:
        with open(args.fastq, 'r') as handle:
            content = handle.readlines()
    reads = [''.join(content[i:i+4]) for i in range(0, len(content), 4)]
    args.reads = {reads[i].split('\n')[0] : reads[i] for i in range(len(reads))}
    del reads
    del content

    # get reads id from reads in tfrecords
    args.reads_id = []
    with open(args.reads_id_file, 'r') as handle:
        for line in handle:
            args.reads_id.append(line.rstrip().split('\t')[0])


def GetAveQualScore(base_qual_scores):
    # convert characters to ASCII code
    int_qual_scores = [ord(c)-33 for c in base_qual_scores]

    # calculate average quality score by first converting Phred scores to probabilities, calculate the average error probability and convert average back to Phred scale
    return -10*math.log(sum([10**(q/-10) for q in int_qual_scores]) / len(int_qual_scores), 10)


def ParseData(args, labels, process_id):
    out_filename = os.path.join(args.output_dir, '-'.join(args.input.split('/')[-1].split('-')[:-1]) + f'-cutoff-{args.cutoff}-{process_id}-taxa_profile')
    labels_count = defaultdict(list)

    with open(args.input, 'r') as f:
        for count, line in enumerate(f, 0):
            if float(line.rstrip().split('\t')[1]) >= args.cutoff:
                if line.rstrip().split('\t')[0] in labels:
                    labels_count[line.rstrip().split('\t')[0]].append(count)
    print('parsing based on confidence score done')
    
    with open(out_filename, 'w') as out_f:
        for label, reads_idx in labels_count.items():
            print(f'# number of reads classified to label {label}: {len(reads_idx)}')
            if args.binning:
                fq_filename = os.path.join(args.output_dir, f'bin-{label}.fq')
                sum_filename = os.path.join(args.output_dir, f'summary-{label}.tsv')
                for idx in reads_idx:
                    print(idx)
                    # get read id
                    read_id = args.reads_id[idx]        
                    # get read based quality score
                    base_qual_scores = args.reads[read_id].split('\n')[3]
                    read_ave_qual_score = GetAveQualScore(base_qual_scores)
                    # get read length
                    read_length = len(args.reads[read_id].split('\n')[1])
                    
                    with open(fq_filename, 'a') as out_fq:
                        out_fq.write(''.join(args.reads[read_id]))

                    with open(sum_filename, 'a') as out_fs:
                        out_fs.write(f'{read_id}\t{read_ave_qual_score}\t{math.ceil(read_ave_qual_score)}\t{read_length}\n')
                    
            out_f.write(f'{label}\t{len(reads_idx)}\t{count+1}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='output file with classification results obtained from running DL-TODA')
    parser.add_argument('--tool', help='type of taxonomic classification tool', choices=['dl-toda', 'kraken2', 'centrifuge'])
    parser.add_argument('--fastq', type=str, help='path to fastq file')
    parser.add_argument('--reads_id_file', type=str, help='path to file containing ordered reads id')
    parser.add_argument('--binning', help='bin reads', action='store_true', required=('--fastq' in sys.argv and '--reads_id' in sys.argv))
    parser.add_argument('--processes', type=int, help='number of processes', default=mp.cpu_count())
    parser.add_argument('--output_dir', type=str, help='path to output directory', default=os.getcwd())
    parser.add_argument('--rank', type=str, help='taxonomic rank at which the analysis should be done', default='species')
    parser.add_argument('--cutoff', type=float, help='cutoff or probability score between 0 and 1 above which reads should be analyzed', default=0.0)
    parser.add_argument('--ncbi_db', help='path to directory containing ncbi taxonomy db')
    parser.add_argument('--labels', nargs='+', default=[], help='list of labels to bin')
    parser.add_argument('--tax_db', help='type of taxonomy database used in DL-TODA', choices=['ncbi', 'gtdb'], default='gtdb')
    parser.add_argument('--summarize', help='summarize taxa profiles from multiple samples', action='store_true')
    parser.add_argument('--class_mapping', type=str, help='path to json file containing dictionary mapping taxa to labels')
    args = parser.parse_args()

    args.ranks = {'phylum': 5, 'class': 4, 'order': 3, 'family': 2, 'genus': 1, 'species': 0}

    # get dl-toda taxonomy
    if args.tax_db == 'ncbi':
        index = 2
    elif args.tax_db =='gtdb':
        index = 1

    if args.summarize:
        input_files = glob.glob(os.path.join(args.input, f'*-taxa_profile'))
        taxa_count = defaultdict(int)
        for i in range(len(input_files)):
            with open(input_files[i], 'r') as f:
                for line in f:
                    if int(line.rstrip().split('\t')[1]) != 0:
                        taxa_count[line.rstrip().split('\t')[0].split(';')[args.ranks[args.rank]]] += int(line.rstrip().split('\t')[1])
        with open(os.path.join(args.output_dir, f'taxa_profile_{args.rank}'), 'w') as out_f:
            for k, v in taxa_count.items():
                out_f.write(f'{k}\t{v}\n')

    if args.binning:
        # get reads from fastq file
        LoadReads(args)

    if args.tool == 'dl-toda':
        if args.class_mapping:
            f = open(args.class_mapping)
            class_mapping = json.load(f)
            args.taxonomy = {k: v.split(';')[0] for k, v in class_mapping.items()} # only get species level
        else:
            # load dl-toda taxonomy
            args.taxonomy = {}
            path_dl_toda_tax = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]) + '/data/dl_toda_taxonomy.tsv'
            with open(path_dl_toda_tax, 'r') as in_f:
                for line in in_f:
                    line = line.rstrip().split('\t')
                    args.taxonomy[str(line[0])] = line[index].split(';')[args.ranks[args.rank]]

        # update list of labels to investigate
        if len(args.labels) != 0:
            labels_to_analyze = args.labels
        else:
            labels_to_analyze = list(args.taxonomy.keys())
        
        # update and create output directory
        args.output_dir = os.path.join(args.output_dir, f'cutoff-{args.cutoff}')
        if not os.path.exists(args.output_dir):
            os.makedirs(os.path.join(args.output_dir))

        # split taxa amongst processes
        chunk_size = math.ceil(len(labels_to_analyze)/args.processes)
        labels_groups = [labels_to_analyze[i:i+chunk_size] for i in range(0,len(labels_to_analyze),chunk_size)]

        with mp.Manager() as manager:
            processes = [mp.Process(target=ParseData, args=(args, labels_groups[i], i)) for i in range(len(labels_groups))]
            for p in processes:
                p.start()
            for p in processes:
                p.join()

    elif args.tool in ['kraken2', 'centrifuge']:
        args.dataset = 'meta'
        args.d_nodes = parse_nodes_file(os.path.join(args.ncbi_db, 'taxonomy', 'nodes.dmp'))
        args.d_names = parse_names_file(os.path.join(args.ncbi_db, 'taxonomy', 'names.dmp'))
        # load results of taxonomic classification tool
        data = load_tool_output(args)
        # parse data
        functions = {'kraken2': parse_kraken_output, 'centrifuge': parse_centrifuge_output}
        with mp.Manager() as manager:
            results = manager.dict()
            processes = [mp.Process(target=functions[args.tool], args=(args, data[i], i, results)) for i in range(len(data))]
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            # combine results from all processes
            taxa_count = defaultdict(int)
            for process, process_results in results.items():
                for i in range(len(process_results)):
                    taxa_count[process_results[i].rstrip().split('\t')[1]] += 1
            # write results to output file
            out_filename = os.path.join(args.output_dir, '-'.join([args.input.split('/')[-1], 'taxa_profile']))
            with open(out_filename, 'w') as out_f:
                for k, v in taxa_count.items():
                    out_f.write(f'{k}\t{v}\n')



##########################################################
        # create file with taxonomic profiles
        # with open(args.output_file, 'w') as out_f:
        #     for k, v in args.dl_toda_taxonomy.items():
        #         num_reads = 0
        #         for process_id, process_results in results.items():
        #             for label, read_count in process_results.items():
        #                 if label == k:
        #                     num_reads += read_count
        #         taxonomy = '\t'.join(v)
        #         out_f.write(f'{taxonomy}\t{num_reads}\n')

                # if args.binning:
                #     # combine fastq files
                #     prefix = k if args.rank == 'species' else v[args.ranks[args.rank]]
                #     fq_files = sorted(glob.glob(os.path.join(args.output_dir, 'tmp', f'{prefix}-*-tmp.fq')))
                #     with open(os.path.join(args.output_dir, f'{prefix}-bin.fq'), 'w') as out_fq:
                #         for fq in fq_files:
                #             with open(fq, 'r') as in_fq:
                #                 out_fq.write(in_fq.read())
        # if args.binning:
        #     # remove tmp fq files
        #     shutil.rmtree(os.path.join(args.output_dir, 'tmp'))
