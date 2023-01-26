import argparse
import sys
import os
import math
import glob
import gzip
import shutil
from collections import defaultdict
import multiprocessing as mp
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
from vis_scripts.parse_tool_output import *
from dataprep_scripts.ncbi_tax_utils import parse_nodes_file, parse_names_file

def load_reads(args):
    if args.fastq[-2:] == 'gz':
        with gzip.open(args.fastq, 'rt') as handle:
            content = handle.readlines()
    else:
        with open(args.fastq, 'r') as handle:
            content = handle.readlines()
    records = [''.join(content[i:i+4]) for i in range(0, len(content), 4)]
    args.reads = {rec.split('\n')[0]: rec for rec in records}

# def parse_data(taxa, data, args, process_id):
def parse_data(taxa, args, process_id):
    labels = [k for k, v in args.dl_toda_taxonomy.items() if v in taxa]
    print(process_id, len(labels))
    out_filename = os.path.join(args.output_dir, '-'.join(args.input.split('/')[-1].split('-')[:-1]) + f'-cutoff-{args.cutoff}-{process_id}-out.tsv')
    taxa_count = defaultdict(int)
    with open(args.input, 'r') as f:
        for line in f:
            if int(line.rstrip().split('\t')[2]) in labels:
                if float(line.rstrip().split('\t')[3]) > args.cutoff:
                    taxa_count[args.dl_toda_taxonomy[int(line.rstrip().split('\t')[2])]] += 1

    # for t in taxa:
    #     # get label(s)
    #     l = [k for k, v in args.dl_toda_taxonomy.items() if v == t]
    #     # get reads
    #     t_reads_id = []
    #     for k, v in data.items():
    #         if int(k) in l:
    #             for i in range(len(v)):
    #                 if float(v[i][3]) > args.cutoff:
    #                     t_reads_id.append(v[i][0])
    #     taxa_count[t] = len(t_reads_id)
        # if args.binning:
        #     t_reads = [args.reads[r] for r in t_reads_id]
        #     fq_filename = os.path.join(args.output_dir, f'{process_id}', f'bin-{l[0]}.fq') if args.rank == 'species' else os.path.join(args.output_dir, f'{process_id}', f'bin-{t.split(";")[0]}.fq')
        #     with open(fq_filename, 'a') as out_fq:
        #         out_fq.write(''.join(t_reads))

    # write tax profile to output file
    with open(out_filename, 'w') as out_f:
        for k, v in taxa_count.items():
            out_f.write(f'{k}\t{v}\n')




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='output file with predicted species obtained from running DL-TODA')
    parser.add_argument('--tool', help='type of taxonomic classification tool', choices=['dl-toda', 'kraken2', 'centrifuge'])
    parser.add_argument('--fastq', type=str, help='path to directory with fastq file', required=('--binning' in sys.argv))
    parser.add_argument('--binning', help='bin reads', action='store_true')
    parser.add_argument('--processes', type=int, help='number of processes', default=mp.cpu_count())
    parser.add_argument('--output_dir', type=str, help='path to output directory', default=os.getcwd())
    parser.add_argument('--rank', type=str, help='taxonomic rank at which the analysis should be done', default='species')
    parser.add_argument('--cutoff', type=float, help='cutoff or probability score between 0 and 1 above which reads should be analyzed', default=0.0)
    parser.add_argument('--ncbi_db', help='path to directory containing ncbi taxonomy db')
    parser.add_argument('--taxa', nargs='+', default=[], help='list of taxa to bin')
    parser.add_argument('--tax_db', help='type of taxonomy database used in DL-TODA', choices=['ncbi', 'gtdb'])
    parser.add_argument('--summarize', help='summarize taxa profiles from multiple samples', action='store_true')
    args = parser.parse_args()

    args.ranks = {'phylum': 5, 'class': 4, 'order': 3, 'family': 2, 'genus': 1, 'species': 0}

    # get dl-toda taxonomy
    if args.tax_db == 'ncbi':
        index = 2
    elif args.tax_db =='gtdb':
        index = 1

    if args.binning:
        # load reads
        load_reads(args)

    elif args.summarize:
        input_files = glob.glob(os.path.join(args.input, '*-taxa_profile'))
        print(len(input_files))
        taxa_count = defaultdict(int)
        for i in range(len(input_files)):
            with open(input_files[i], 'r') as f:
                for line in f:
                    taxa_count[line.rstrip().split('\t')[0].split(';')[args.ranks[args.rank]]] += int(line.rstrip().split('\t')[1])
        with open(os.path.join(args.output_dir, f'{args.rank}-taxa_profile'), 'w') as out_f:
            for k, v in taxa_count.items():
                out_f.write(f'{k}\t{v}\n')

    if args.tool == 'dl-toda':
        # load dl-toda taxonomy
        args.dl_toda_taxonomy = {}
        with open('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]) + '/data/dl_toda_taxonomy.tsv', 'r') as in_f:
            for line in in_f:
                line = line.rstrip().split('\t')
                args.dl_toda_taxonomy[int(line[0])] = ';'.join(line[index].split(';')[args.ranks[args.rank]:])
        taxa = []
        for i in range(len(args.dl_toda_taxonomy)):
            if args.dl_toda_taxonomy[i] not in taxa:
                taxa.append(args.dl_toda_taxonomy[i])
        print(len(taxa))
        # update and create output directory
        args.output_dir = os.path.join(args.output_dir, '-'.join(args.input.split('/')[-1].split('-')[:-1]), f'cutoff-{args.cutoff}')
        if not os.path.exists(args.output_dir):
            os.makedirs(os.path.join(args.output_dir))
            if args.binning:
                for i in range(args.processes):
                    os.makedirs(os.path.join(args.output_dir, f'{i}'))

        # split taxa amongst processes
        chunk_size = math.ceil(len(taxa)/args.processes)
        taxa_groups = [taxa[i:i+chunk_size] for i in range(0,len(taxa),chunk_size)]
        print(chunk_size, len(taxa_groups), len(taxa_groups[0]))

        # load data
        # data = defaultdict(list)
        # with open(args.dl_toda_output, 'r') as f:
        #     for line in f:
        #         data[line.rstrip().split('\t')[2]].append(line.rstrip().split('\t'))
        # print(len(data))
        #
        # data_toshare = [data[i] for i in taxa_groups[0]]
        # print(len(data_toshare))
        # print(len(taxa_groups[0]))

        with mp.Manager() as manager:
            processes = [mp.Process(target=parse_data, args=(taxa_groups[i], args, i)) for i in range(len(taxa_groups))]
            # processes = [mp.Process(target=parse_data, args=(taxa_groups[i], [data[j] for j in taxa_groups[i]], args, i)) for i in range(len(taxa_groups))]
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
