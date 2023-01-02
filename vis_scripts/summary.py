import argparse
import os
import sys
import pandas as pd
import glob
import numpy as np
import multiprocessing as mp
from collections import defaultdict
from parse_tool_output import *
from ncbi_tax_utils import parse_nodes_file, parse_names_file
from summary_utils import *

def create_cm(args):
    # load cami ground truth if testing set was made by cami
    if args.dataset == 'cami':
        args.cami_data = load_cami_data(args)
    # if args.dataset == 'cami' or args.tool in ['kraken', 'centrifuge']:
    # load results of taxonomic classification tool
    data = load_tool_output(args)
    # parse data
    functions = {'kraken': parse_kraken_output, 'dl-toda': parse_dl_toda_output, 'centrifuge': parse_centrifuge_output}
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
        for r_name, r_index in args.ranks.items():
            cm = fill_out_cm(args, predictions, ground_truth, confidence_scores, r_name, r_index)
            # store confusion matrices in excel file
            with pd.ExcelWriter(os.path.join(args.output_dir, f'{args.input.split("/")[-1]}-cutoff-{args.cutoff}-{r_name}-confusion-matrix.xlsx')) as writer:
                cm.to_excel(writer, sheet_name=f'{r_name}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='taxonomic classification tool output file or confusion matrix excel file')
    parser.add_argument('--tool', type=str, help='taxonomic classification tool', choices=['kraken', 'dl-toda', 'centrifuge'])
    parser.add_argument('--dataset', type=str, help='dataset ground truth', choices=['cami', 'testing', 'meta'])
    parser.add_argument('--cutoff', type=float, help='decision thershold above which reads are classified', default=0.0)
    parser.add_argument('--combine', help='summarized results from all samples combined', action='store_true', required=('--input_dir' in sys.argv))
    parser.add_argument('--metrics', help='get metrics from confusion matrix', action='store_true')
    parser.add_argument('--confusion_matrix', help='create confusion matrix', action='store_true')
    parser.add_argument('--tax', help='get full taxonomy from predicted labels', action='store_true')
    parser.add_argument('--zeros', help='add ground truth taxa with a null precision, recall and F1 metrics', action='store_true')
    parser.add_argument('--unclassified', help='add unclassified reads to the calculation of recall', action='store_true')
    parser.add_argument('--input_dir', type=str, help='path to input directory containing excel files to combine', default=os.getcwd())
    parser.add_argument('--output_dir', type=str, help='path to output directory', default=os.getcwd())
    parser.add_argument('--dl_toda_tax', help='path to directory containing json directories with info on taxa present in dl-toda')
    parser.add_argument('--tax_db', help='type of taxonomy database used in DL-TODA', choices=['ncbi', 'gtdb'])
    parser.add_argument('--ncbi_db', help='path to directory containing ncbi taxonomy db')

    # parser.add_argument('--to_ncbi', action='store_true', help='whether to analyze results with ncbi taxonomy', default=False)
    # parser.add_argument('--compare', action='store_true', help='compare results files obtained with --metrics', default=False)
    # parser.add_argument('--rank', type=str, help='taxonomic rank', choices=['species', 'genus', 'family', 'order', 'class', 'phylum'], required=('--compare' in sys.argv))
    # parser.add_argument('--dl_toda_metrics', type=str, help='path to result file obtained from running --metrics', required=('--compare' in sys.argv))
    # parser.add_argument('--tool_metrics', type=str, help='path to result file obtained from running --metrics with another tool than DL-TODA', required=('--compare' in sys.argv))


    args = parser.parse_args()

    args.ranks = {'species': 0, 'genus': 1, 'family': 2, 'order': 3, 'class': 4, 'phylum': 5}

    # load dl-toda ground truth taxonomy
    if args.dl_toda_tax:
        index = 1 if args.tax_db == "gtdb" else 2
        with open(os.path.join(args.dl_toda_tax, 'dl_toda_taxonomy.tsv'), 'r') as in_f:
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

            with pd.ExcelWriter(os.path.join(args.output_dir, f'confusion-matrix.xlsx')) as writer:
                for r_name, r_cm in all_cm.items():
                    r_cm.to_excel(writer, sheet_name=f'{r_name}')

    if args.metrics:
        cm = pd.read_excel(args.input, index_col=0, sheet_name=None)
        for r_name, r_index in args.ranks.items():
            if r_name in cm.keys():
                get_metrics(args, cm[r_name], r_name, r_index)

    if args.tax:
        # load dl-toda results
        data = load_tool_output(args)
        with mp.Manager() as manager:
            results = manager.dict()
            processes = [mp.Process(target=parse_dl_toda_output, args=(args, data[i], i, d_nodes, d_names, results)) for i in range(len(data))]
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            # write results to file
            out_filename = args.input[:-4] + '-tax.tsv'
            with open(out_filename, 'w') as out_f:
                for process, process_results in results.items():
                    for i in range(len(process_results)):
                        out_f.write(f'{process_results[i][0]}\t{process_results[i][2]}\n')



    # elif args.compare:
    #     dl_toda_res = pd.read_csv(args.dl_toda_metrics, sep='\t')
    #     tool_res = pd.read_csv(args.tool_metrics, sep='\t')
    #     dl_toda_true_taxa = dl_toda_res['true taxon'][:-4].tolist()
    #
    #     # check data
    #     if dl_toda_true_taxa.sort() != tool_res['true taxon'][:-4].tolist().sort():
    #         sys.exit(f'Not the same taxa between DL-TODA and tool at rank: {args.rank}')
    #     if len(dl_toda_true_taxa) != len(tool_res['true taxon'][:-4].tolist()):
    #         sys.exit(f'Not the same number of taxa between DL-TODA and tool at rank: {args.rank}')
    #     if dl_toda_res['#reads'][:-4].tolist().sort() != tool_res['#reads'][:-4].tolist().sort():
    #         sys.exit(f'Not the same #reads column between DL-TODA and tool at rank: {args.rank}')
    #
    #     # compute percentage difference between true positives for DL-TODA and the other tool
    #     diff = [((dl_toda_res['TP'][:-4][i]-tool_res['TP'][:-4][i])/dl_toda_res['#reads'][:-4][i])*100 for i in range(len(dl_toda_true_taxa))]
    #     if len(diff) != len(dl_toda_true_taxa):
    #         sys.exit(f'Issue with computing percentage difference at rank: {args.rank}')
    #
    #     with open(os.path.join(args.output_dir, f'DL-TODA-vs-Kraken2-{args.dataset}-set-{args.rank}.tsv'), 'w') as out_f:
    #         for i in range(len(dl_toda_true_taxa)):
    #             out_f.write(f'{dl_toda_true_taxa[i]}\t{dl_toda_res["#reads"][:-4][i]}\t{dl_toda_res["TP"][:-4][i]}\t{tool_res["TP"][:-4][i]}\t{diff[i]}\n')
    #
    #     low_out_f = open(os.path.join(args.output_dir, f'{args.rank}-lower-performance.tsv'), 'w')
    #     high_out_f = open(os.path.join(args.output_dir, f'{args.rank}-higher-performance.tsv'), 'w')
    #
    #     for i in range(len(dl_toda_true_taxa)):
    #         if dl_toda_res['TP'][:-4][i] > tool_res['TP'][:-4][i]:
    #             high_out_f.write(f'{dl_toda_true_taxa[i]}\t{dl_toda_res["TP"][:-4][i]}\t{tool_res["TP"][:-4][i]}\n')
    #         else:
    #             low_out_f.write(f'{dl_toda_true_taxa[i]}\t{dl_toda_res["TP"][:-4][i]}\t{tool_res["TP"][:-4][i]}\n')





if __name__ == "__main__":
    main()
