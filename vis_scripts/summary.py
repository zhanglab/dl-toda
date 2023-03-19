import argparse
import os
import sys
import pandas as pd
import glob
import numpy as np
import statistics
import multiprocessing as mp
from collections import defaultdict
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
from parse_tool_output import *
from dataprep_scripts.ncbi_tax_utils import parse_nodes_file, parse_names_file
from summary_utils import *


def create_cm(args):
    # load cami ground truth if testing set was made by cami
    if args.dataset == 'cami':
        args.cami_data = load_cami_data(args)
    # if args.dataset == 'cami' or args.tool in ['kraken', 'centrifuge']:
    # load results of taxonomic classification tool
    data = load_tool_output(args)
    # parse data
    functions = {'kraken': parse_kraken_output, 'dl-toda': parse_dl_toda_output, 'centrifuge': parse_centrifuge_output, 'bertax': parse_bertax_output}
    with mp.Manager() as manager:
        results = manager.dict()
        processes = [mp.Process(target=functions[args.tool], args=(args, data[i], i, results)) for i in range(len(data))]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        # combine results from all processes
        predictions = []
        ground_truth = []
        confidence_scores = []
        for process, process_results in results.items():
            predictions += [i[0] for i in process_results]
            ground_truth += [i[1] for i in process_results]
            confidence_scores += [i[2] for i in process_results]
        # create confusion matrix
        if args.tool == 'dl-toda':
            for r_name, r_index in args.ranks.items():
                cm = fill_out_cm(args, predictions, ground_truth, confidence_scores, r_index)
                # store confusion matrices in excel file
                with pd.ExcelWriter(os.path.join(args.output_dir,
                                                 f'{args.input.split("/")[-1]}-cutoff-{args.cutoff}-{r_name}-confusion-matrix.xlsx')) as writer:
                    cm.to_excel(writer, sheet_name=f'{r_name}')
        elif args.tool == 'bertax':
            cm = fill_out_cm(args, predictions, ground_truth, confidence_scores, 1)
            # store confusion matrices in excel file
            with pd.ExcelWriter(os.path.join(args.output_dir,
                                             f'{args.input.split("/")[-1]}-cutoff-{args.cutoff}-genus-confusion-matrix.xlsx')) as writer:
                cm.to_excel(writer, sheet_name=f'genus')




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='taxonomic classification tool output file or confusion matrix excel file')
    parser.add_argument('--tool', type=str, help='taxonomic classification tool', choices=['kraken', 'dl-toda', 'centrifuge', 'bertax'])
    parser.add_argument('--dataset', type=str, help='dataset ground truth', choices=['cami', 'testing', 'meta'])
    parser.add_argument('--cutoff', type=float, help='decision thershold above which reads are classified', default=0.0)
    parser.add_argument('--combine', help='summarized results from all samples combined', action='store_true', required=('--input_dir' in sys.argv))
    parser.add_argument('--metrics', help='get metrics from confusion matrix', action='store_true')
    parser.add_argument('--confusion_matrix', help='create confusion matrix', action='store_true')
    parser.add_argument('--probs', help='analysis of probability scores', action='store_true')
    parser.add_argument('--zeros', help='add ground truth taxa with a null precision, recall and F1 metrics', action='store_true')
    parser.add_argument('--unclassified', help='add unclassified reads to the calculation of recall', action='store_true')
    parser.add_argument('--input_dir', type=str, help='path to input directory containing excel files to combine', default=os.getcwd())
    parser.add_argument('--output_dir', type=str, help='path to output directory', default=os.getcwd())
    parser.add_argument('--tax_db', help='type of taxonomy database used in DL-TODA', choices=['ncbi', 'gtdb'])
    parser.add_argument('--ncbi_db', help='path to directory containing ncbi taxonomy db')
    parser.add_argument('--roc', help='option to generate decision thresholds with ROC curves', action='store_true')

    args = parser.parse_args()

    args.ranks = {'species': 0, 'genus': 1, 'family': 2, 'order': 3, 'class': 4, 'phylum': 5}

    # load dl-toda ground truth taxonomy
    if args.dataset == 'testing':
        index = 1 if args.tax_db == "gtdb" else 2
        path_dl_toda_tax = '/'.join(
            os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]) + '/data/dl_toda_taxonomy.tsv'
        with open(path_dl_toda_tax, 'r') as in_f:
            content = in_f.readlines()
            args.dl_toda_tax = {line.rstrip().split('\t')[0]: line.rstrip().split('\t')[index] for line in content}

    # get ncbi taxids info
    if args.ncbi_db:
        args.d_nodes = parse_nodes_file(os.path.join(args.ncbi_db, 'taxonomy', 'nodes.dmp'))
        args.d_names = parse_names_file(os.path.join(args.ncbi_db, 'taxonomy', 'names.dmp'))

    if args.confusion_matrix:
        # create confusion matrix
        create_cm(args)

    if args.combine:
        with mp.Manager() as manager:
            all_cm = manager.dict()
            # Create new processes
            processes = [mp.Process(target=combine_cm, args=(args, all_cm, r)) for r in args.ranks.keys()]
            for p in processes:
                p.start()
            for p in processes:
                p.join()

            with pd.ExcelWriter(os.path.join(args.output_dir, f'cm.xlsx')) as writer:
                for r_name, r_cm in all_cm.items():
                    r_cm.to_excel(writer, sheet_name=f'{r_name}')

        files_to_rm = glob.glob(os.path.join(args.input_dir, f'*-confusion-matrix.xlsx'))
        for f in files_to_rm:
            os.remove(f)

    if args.metrics:
        cm = pd.read_excel(args.input, index_col=0, sheet_name=None)
        for r_name, r_index in args.ranks.items():
            if r_name in cm.keys():
                get_metrics(args, cm[r_name], r_name, r_index)

    if args.probs:
        # load dl-toda results
        data = load_tool_output(args)
        with mp.Manager() as manager:
            results = manager.dict()
            processes = [mp.Process(target=parse_dl_toda_output, args=(args, data[i], i, results)) for i in range(len(data))]
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            # write results to file
            stats_file = open(args.input[:-4] + '-stats.tsv', 'w')
            conf_scores_file = open(args.input[:-4] + '-cs.tsv', 'w')
            scores = []
            for process, process_results in results.items():
                for i in range(len(process_results)):
                    scores.append(process_results[i][2])
                    for k, v in args.ranks.items():
                        conf_scores_file.write(f'{process_results[i][0].split(";")[v]}\t{process_results[i][1].split(";")[v]}\t')
                    conf_scores_file.write(f'{process_results[i][2]}\n')
            stats_file.write(f'{args.input.split("/")[-1][:-8]}\t{statistics.mean(scores)}\t{statistics.median(scores)}'
                             f'\t{min(scores)}\t{max(scores)}\t{len(scores)}\n')


if __name__ == "__main__":
    main()
