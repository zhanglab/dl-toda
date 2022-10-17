import sys
import os
import pandas as pd
from collections import defaultdict
import json

def parse_nodes_file(nodes_file):
    data = {}
    df = pd.read_csv(nodes_file, delimiter='\t', low_memory=False, header=None)
    list_taxids = df.iloc[:, 0].tolist()
    list_parents = df.iloc[:, 2].tolist()
    list_ranks = df.iloc[:, 4].tolist()
    for i in range(len(list_taxids)):
        data[str(list_taxids[i])] = [str(list_parents[i]), list_ranks[i]]
    return data

def parse_names_file(names_file):
    data = defaultdict(dict)
    df = pd.read_csv(names_file, delimiter='\t', low_memory=False, header=None)
    list_taxids = df.iloc[:, 0].tolist()
    list_names = df.iloc[:, 2].tolist()
    list_name_types = df.iloc[:, 6].tolist()
    for i in range(len(list_taxids)):
        if list_name_types[i] == 'scientific name' or list_name_types[i] == 'includes' or list_name_types[i] == 'synonym':
            if list_name_types[i] not in data[str(list_taxids[i])]:
                data[str(list_taxids[i])][list_name_types[i]] = {list_names[i]}
            else:
                data[str(list_taxids[i])][list_name_types[i]].add(list_names[i])

    return data

def get_ncbi_taxonomy(current_taxid, d_nodes, d_names):
    ranks = {'strain': 0, 'species': 1, 'genus': 2, 'family': 3, 'order': 4, 'class': 5, 'phylum': 6}

    # initialize list of taxids, taxa and ranks
    list_taxids = ['na']*len(ranks)
    list_taxa = ['na']*len(ranks)
    list_ranks = ['na']*len(ranks)

    # initialize parent to current taxid
    parent = str(current_taxid)

    while parent != '1' and parent != '0':
        t_rank = d_nodes[parent][1]
        if t_rank in ranks:
            # get name of taxon
            if parent in d_names:
                if 'scientific name' in d_names[parent].keys():
                    if len(d_names[parent]['scientific name']) > 1:
                        sys.exit(f'more than one scientific names: {parent}\t{current_taxid}\t{d_names[parent]["scientific name"]}')
                    else:
                        t_name = list(d_names[parent]['scientific name'])[0]
                else:
                    sys.exit(f'scientific name not in names type: {parent}\t{current_taxid}')
                    # t_name = d_names[parent]['includes'] if 'includes' in d_names[parent] else d_names[parent]['synonym']
            else:
                print(f'{parent} not in ncbi tax db')
                t_name = 'na'
            if t_name == '':
                t_name = 'na'

            list_taxa[ranks[t_rank]] = t_name
            list_taxids[ranks[t_rank]] = parent
            list_ranks[ranks[t_rank]] = t_rank

        parent = d_nodes[parent][0]

    return '|'.join(list_taxids), list_taxa, ';'.join(list_ranks)
