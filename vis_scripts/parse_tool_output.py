import os
import gzip
import math
import multiprocessing as mp
from collections import defaultdict
from ncbi_tax_utils import get_ncbi_taxonomy

def parse_kraken_output(args, data, process, d_nodes, d_names, results):
    process_results = []
    for line in data:
        line = line.rstrip().split('\t')
        read = line[1]
        if args.dataset == 'meta':
            if line[0] == 'U':
                process_results.append(f'{read}\t{";".join(["unclassified"]*7)}\n')
            else:
                # get ncbi taxid of predicted taxon
                taxid_index = line[2].find('taxid')
                taxid = line[2][taxid_index+6:-1]
                # get ncbi taxonomy
                pred_taxonomy = get_ncbi_taxonomy(taxid, d_nodes, d_names)
                process_results.append(f'{read}\t{pred_taxonomy}\n')
        else:
            if args.dataset == 'cami':
                true_taxonomy = get_ncbi_taxonomy(args.cami_data[read], d_nodes, d_names)
            else:
                true_taxonomy = get_dl_toda_taxonomy(args, read.split('|')[1])
            if line[0] == 'U':
                process_results.append(f'{read}\t{";".join(["unclassified"]*7)}\t{true_taxonomy}\n')
            else:
                # get ncbi taxid of predicted taxon
                taxid_index = line[2].find('taxid')
                taxid = line[2][taxid_index+6:-1]
                # get ncbi taxonomy
                pred_taxonomy = get_ncbi_taxonomy(taxid, d_nodes, d_names)
                process_results.append(f'{read}\t{pred_taxonomy}\t{true_taxonomy}\n')

    results[process] = process_results

def parse_dl_toda_output(args, data, process, d_nodes, d_names, results):
    process_results = []
    for line in data:
        read_id = line.rstrip().split('\t')[0]
        pred_sp = line.rstrip().split('\t')[1]
        confidence_score = float(line.rstrip().split('\t')[2])
        pred_taxonomy = args.dl_toda_tax[pred_sp]
        if args.dataset == 'cami':
            true_taxonomy = get_ncbi_taxonomy(args.cami_data[read_id], d_nodes, d_names)
        else:
            true_taxonomy = args.dl_toda_tax[read_id.split('|')[1]]
        process_results.append([pred_taxonomy, true_taxonomy, confidence_score])
    results[process] = process_results
    print(process, len(process_results), len(results[process]))

def parse_centrifuge_output(args, data, process, d_nodes, d_names, results):
    # centrifuge output shows multiple possible hits per read, choose hit with best score (first hit)
    process_results = []
    number_unclassified = 0
    for line in data:
        read = line.rstrip().split('\t')[0]
        taxid = line.rstrip().split('\t')[2]
        if args.dataset == 'cami':
            true_taxonomy = get_ncbi_taxonomy(args.cami_data[read], d_nodes, d_names)
        else:
            true_taxonomy = get_dl_toda_taxonomy(args, read.split('|')[1])
        if taxid != '0':
            pred_taxonomy = get_ncbi_taxonomy(taxid, d_nodes, d_names)
            process_results.append(f'{read}\t{pred_taxonomy}\t{true_taxonomy}\n')
        else:
            process_results.append(f'{read}\t{";".join(["unclassified"]*7)}\t{true_taxonomy}\n')
            number_unclassified += 1

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

    return data
