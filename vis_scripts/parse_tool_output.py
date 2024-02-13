import os
import sys
import gzip
import math
import numpy as np
import multiprocessing as mp
from collections import defaultdict
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
from dataprep_scripts.ncbi_tax_utils import get_ncbi_taxonomy
from dataprep_scripts.utils import load_fq_file


def parse_bert_output(args, data, process, results):
    """ Bert output contains the probability distributions for each example (one example per line)
    """
    # get labels from fastq file
    reads = load_fq_file(args.fq_file, 4)
    labels = [args.labels_mapping[i.split('\n')[0].split('|')[1]] for i in reads]
    print(labels[:10])

    process_results = []
    for i, line in enumerate(data, 0):
        probs = [float(v) for v in line.rstrip().split('\t')]
        print(probs)
        true_taxonomy = args.dl_toda_tax[labels[i]]
        # get highest probability
        confidence_score = np.amax(probs)
        print(f'confidence_score: {confidence_score}')
        print(f'np.argmax(probs): {np.argmax(probs)}')
        pred_taxonomy = args.dl_toda_tax[str(np.argmax(probs))]
        print(f'pred_taxonomy: {pred_taxonomy}')
        process_results.append([pred_taxonomy, true_taxonomy, confidence_score])

    results[process] = process_results


def parse_bertax_output(args, data, process, results):
    """ Bertax output is provided at the genus, phylum and superkingdom levels
        Reads assigned to the Viruses or Eukaryota are considered as unclassified
        All other reads are considered classified regardless of confidence scores
    """

    process_results = []
    for line in data:
        line = line.rstrip().split('\t')
        read = line[0]
        # get ground truth
        true_taxonomy = args.dl_toda_tax[read.split('|')[1]]
        confidence_score = int(line[6][:-1]) / 100
        if line[1] == 'Bacteria' and line[5] != 'unknown':
            pred_taxonomy = f';{line[5]};;;;'
        else:
            pred_taxonomy = f';unclassified;;;;'
        process_results.append([pred_taxonomy, true_taxonomy, confidence_score])

    results[process] = process_results


def parse_kraken_output(args, data, process, results):
    process_results = []
    for line in data:
        line = line.rstrip().split('\t')
        read = line[1]
        if args.dataset == 'meta':
            if line[0] != 'U':
                # get ncbi taxid of predicted taxon
                taxid_index = line[2].find('taxid')
                taxid = line[2][taxid_index+6:-1]
                # get ncbi taxonomy
                _, pred_taxonomy, _ = get_ncbi_taxonomy(taxid, args.d_nodes, args.d_names)
                process_results.append(f'{read}\t{";".join(pred_taxonomy[args.ranks[args.rank]+1:])}\n')
        else:
            if args.dataset == 'cami':
                true_taxonomy = get_ncbi_taxonomy(args.cami_data[read], args.d_nodes, args.d_names)
            else:
                true_taxonomy = get_dl_toda_taxonomy(args, read.split('|')[1])
            if line[0] == 'U':
                process_results.append(f'{read}\t{";".join(["unclassified"]*7)}\t{true_taxonomy}\n')
            else:
                # get ncbi taxid of predicted taxon
                taxid_index = line[2].find('taxid')
                taxid = line[2][taxid_index+6:-1]
                # get ncbi taxonomy
                pred_taxonomy = get_ncbi_taxonomy(taxid, args.d_nodes, args.d_names)
                process_results.append(f'{read}\t{pred_taxonomy}\t{true_taxonomy}\n')

    results[process] = process_results


def parse_dl_toda_output(args, data, process, results):
    process_results = []
    for line in data:
        read_id = line.rstrip().split('\t')[0]
        if args.dataset == 'cami':
            pred_sp = line.rstrip().split('\t')[2]
            confidence_score = float(line.rstrip().split('\t')[3])
            true_taxonomy = get_ncbi_taxonomy(args.cami_data[read_id], args.d_nodes, args.d_names)
        elif args.dataset == 'testing':
            pred_sp = line.rstrip().split('\t')[1]
            confidence_score = float(line.rstrip().split('\t')[2])
            true_taxonomy = args.dl_toda_tax[line.rstrip().split('\t')[0]]
        elif args.dataset == 'meta':
            pred_sp = line.rstrip().split('\t')[2]
            confidence_score = float(line.rstrip().split('\t')[3])
            true_taxonomy = 'na'
        if confidence_score > args.cutoff:
            pred_taxonomy = args.dl_toda_tax[pred_sp]
        else:
            pred_taxonomy = ";".join(["unclassified"]*6)
        process_results.append([pred_taxonomy, true_taxonomy, confidence_score])
    results[process] = process_results


def parse_centrifuge_output(args, data, process, results):
    # centrifuge output shows multiple possible hits per read, choose hit with best score (first hit)
    process_results = []
    read_count = {"strain": 0, "species": 0, "genus": 0, "family": 0, "order": 0, "class": 0, "phylum": 0, "superkingdom": 0}
    num_unclassified = 0
    for line in data:
        read = line.rstrip().split('\t')[0]
        taxid = line.rstrip().split('\t')[2]
        rank = line.rstrip().split('\t')[1]

        if rank not in read_count.keys():
            read_count['strain'] += 1
        else:
            read_count[rank] += 1

        if taxid != '0':
            _, pred_taxonomy, _ = get_ncbi_taxonomy(taxid, args.d_nodes, args.d_names)
            # update predicted taxonomy based on rank of prediction
            if rank in ["genus", "family", "order", "class", "phylum"]:
                pred_taxonomy = ["unclassified"]*(args.ranks[rank]) + pred_taxonomy[args.ranks[rank]+1:]
            elif rank == 'superkingdom':
                pred_taxonomy = ["unclassified"]*6
            else:
                pred_taxonomy = pred_taxonomy[1:]
            if args.dataset == 'meta':
                process_results.append(f'{read}\t{";".join(pred_taxonomy)}\n')
            else:
                if args.dataset == 'cami':
                    true_taxonomy = get_ncbi_taxonomy(args.cami_data[read], args.d_nodes, args.d_names)
                else:
                    true_taxonomy = get_dl_toda_taxonomy(args, read.split('|')[1])
                    process_results.append(f'{read}\t{pred_taxonomy}\t{true_taxonomy}\n')

        elif taxid == '0' and args.dataset != 'meta':
            process_results.append(f'{read}\t{";".join(["unclassified"]*7)}\t{true_taxonomy}\n')

        elif taxid == '0' and args.dataset == 'meta':
            num_unclassified += 1
            process_results.append(f'{read}\t{";".join(["unclassified"] * 6)}\n')
    # print(read_count)
    # print(num_unclassified)
    results[process] = process_results

# def convert_diamond_output(args, data, process, d_nodes, d_names, results):
#     # centrifuge output shows multiple possible hits per read, choose hit with best score (first hit)
#     process_results = []
#     number_unclassified = 0
#     for line in data:
#         read = line.rstrip().split('\t')[0]
#         taxid = line.rstrip().split('\t')[-1]
#         if args.dataset == 'cami':
#             true_taxonomy = get_ncbi_taxonomy(args.cami_data[read], d_nodes, d_names)
#         else:
#             true_taxonomy = get_dl_toda_taxonomy(args, read.split('|')[1])
#         if taxid != '0':
#             pred_taxonomy = get_ncbi_taxonomy(taxid, d_nodes, d_names)
#             process_results.append(f'{read}\t{pred_taxonomy}\t{true_taxonomy}\n')
#         else:
#             process_results.append(f'{read}\t{";".join(["unclassified"]*7)}\t{true_taxonomy}\n')
#             number_unclassified += 1
#     results[process] = process_results

# returns a dictionary with gene ID as key and the taxonomy as the value
# def parse_metaphlan_database(file_path):
#     parse_dict = {}
#     with open(file_path, 'r') as f:
#         for line in f:
#             if line[0] == '>':
#                 temp = line.split("\t")
#                 if len(temp) < 2:
#                     taxonomy = 'na'
#                 else:
#                     temp[1] = taxonomy = taxonomy.split(';')
#                     taxonomy = taxonomy[1][::-1]
#
#                 parse_dict[temp[0]] = 'na' + taxonomy
#     return parse_dict

# def convert_metaphlan_output(args, data, process, db_dict, results):
#     process_results = []
#     for line in data:
#         read = line.rstrip().split('\t')[0]
#         gene_id = line.rstrip().split('\t')[1]
#         if args.dataset == 'cami':
#             true_taxonomy = get_ncbi_taxonomy(args.cami_data[read], db_dict)
#         else:
#             true_taxonomy = get_dl_toda_taxonomy(args, read.split('|')[1])
#         if taxid != '0':
#             pred_taxonomy = db_dict[gene_id]
#             process_results.append(f'{read}\t{pred_taxonomy}\t{true_taxonomy}\n')
#         else:
#             process_results.append(f'{read}\t{";".join(["unclassified"] * 7)}\t{true_taxonomy}\n')
#     results[process] = process_results


def load_cami_data(args):
    in_f = gzip.open(os.path.join(args.cami_path, 'reads_mapping.tsv.gz'), 'rb')
    content = in_f.readlines()
    # create dictionary mapping reads id to ncbi taxid
    data = {line.decode('utf8').rstrip().split('\t')[0]: line.decode('utf8').rstrip().split('\t')[2] for line in content[1:]}

    return data


def load_tool_output(args):
    in_f = open(args.input, 'r')
    content = in_f.readlines()
    if args.tool == 'bertax':
        content = content[1:]
    if args.tool == "centrifuge":
        # parse output of centrifuge to only take the first hit for each read
        content = content[1:]
        # reads_seen = set()
        parsed_content = defaultdict(list)
        for line in content:
            read = line.rstrip().split('\t')[0]
            parsed_content[read].append(line)
            # if read not in reads_seen:
                # parsed_content.append(line)
                # reads_seen.add(read)
        content = [v[0] if len(v) == 1 else '\t'.join([v[0].rstrip().split('\t')[0], '' ,'0']) for k, v in parsed_content.items()]
    if args.tool == 'diamond':
        content = content[3:]
        reads_seen = set()
        parsed_content = []
        for line in content:
            read = line.rstrip().split('\t')[0]
            if read not in reads_seen:
                parsed_content.append(line)
                reads_seen.add(read)
        content = parsed_content

    # get length of sub-arrays
    chunk_size = math.ceil(len(content)/mp.cpu_count())
    data = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
    num_reads = [len(i) for i in data]
    print(chunk_size, len(data), num_reads)
    print(args.dl_toda_tax)

    return data
