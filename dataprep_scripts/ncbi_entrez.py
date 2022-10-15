from Bio import Entrez
import sys
import os
from ncbi_tax_utils import get_ncbi_taxonomy, parse_nodes_file, parse_names_file
import argparse

def retrieve_sp_taxid(list_uids):
    """ returns a list of species ncbi taxids for every ncbi unique identifier in the input list """
    request = Entrez.esummary(db="assembly", id=",".join(list_uids), rettype="uilist", retmode="text")
    result = Entrez.read(request, validate=False)
    request.close()
    print(result['DocumentSummarySet']['DocumentSummary'])
    list_species_taxid = [i['SpeciesTaxid'] for i in result['DocumentSummarySet']['DocumentSummary']]
    # list_strain_taxid = [i['Taxid'] for i in result['DocumentSummarySet']['DocumentSummary']]
    return list_species_taxid

def retrieve_uid(genome_id):
    """ returns ncbi unique identifier of given genome accession id"""
    request = Entrez.esearch(db="assembly", retmax=10, term=genome_id, id_type="acc")
    result = Entrez.read(request)
    request.close()
    if int(result["Count"]) > 1:
        sys.exit(-1)
    else:
        uid = result["IdList"][0]

    return uid

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--email', type=str, help='email required to access NCBI Entrez')
    parser.add_argument('--output_dir', type=str, help='path to output directory')
    parser.add_argument('--genomes', type=str, help='path to tab separated file containing list of genomes id')
    parser.add_argument('--ncbi_db', type=str, help='path to directory containing nodes.dmp and names.dmp NCBI taxonomy files')
    parser.add_argument('--taxonomy', help='taxonomy of genome', action='store_true', required=('--ncbi_db' in sys.argv))
    args = parser.parse_args()

    Entrez.email = args.email

    # get genomes
    with open(args.genomes, 'r') as f:
        content = f.readlines()
        genomes = [i.rstrip().split('\t')[0] for i in content]

    # get genomes Entrez Unique Identifiers
    list_uids = []
    for genome_id in genomes:
        list_uids.append(retrieve_uid(genome_id))

    # get ncbi taxonomy
    if args.taxonomy:
        # parse nodes.dmp and names.dmp ncbi taxonomy files
        d_nodes = parse_nodes_file(os.path.join(args.ncbi_db, 'nodes.dmp'))
        d_names = parse_names_file(os.path.join(args.ncbi_db, 'names.dmp'))

        list_species_taxid = retrieve_sp_taxid(list_uids)
        with open(os.path.join(args.output_dir, 'genomes-ncbi-taxonomy.tsv'), 'w') as out_f:
            for i in range(len(list_species_taxid)):
                list_taxids, list_taxa, _ = get_ncbi_taxonomy(list_species_taxid[i], d_nodes, d_names)
                out_f.write(f'{genomes[i]}')
                for j in range(1,len(list_taxa),1):
                    out_f.write(f'\t{list_taxa[j]}')
                out_f.write(f'\t{list_taxids}')
                out_f.write(f'\t{list_uids[i]}\n')


if __name__ == "__main__":
    main()
