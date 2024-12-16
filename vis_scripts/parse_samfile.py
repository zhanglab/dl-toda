import sys
import os
from collections import defaultdict
import math
import argparse

# symbols in CIGAR string
# M: match, no insertion or deletions, bases may not agree --> consumes query and ref
# I: insertion, additional base in query (not in reference) --> consumes query but not ref
# D: deletion, query is missing base from reference --> consumes ref but not query
# =: equal, no insertions or deletions, and bases agree --> consumes query and ref
# X: not equal, no insertions or deletions, bases do not agree --> consumes query and ref
# N: none, no query bases to align, an expected read gap (spliced read) --> consumes ref but not query
# S: soft-clipped, bases on end of read are not aligned but stored in SAM --> consumes query but not ref
# H: hard-clipped, bases on end of read are not aligned, not stored in SAM --> does no consume query or ref
# P: padding, neither read nor reference has a base here --> does no consume query or ref

refmoveset = {'M', '=', 'X', 'D', 'N'}
refnomoveset = {'I', 'S', 'H', 'P'}


def ExtendCigar(cigar):
    new_cigar = ''
    num = ''
    for i in cigar:
        if i.isnumeric():
            num += i
        else:
            new_cigar += i*int(num)
            num = ''
    return new_cigar


def GetCoverage(list_of_reads, length_ref):
    dict_coverage = {i: 0 for i in range(length_ref)}
    reads_info = defaultdict(list)

    for j in range(0, len(list_of_reads)):
        read_id = list_of_reads[j][0]
        read_start = list_of_reads[j][1] - 1 
        read_cigar = ExtendCigar(list_of_reads[j][2])
        query_pos = 0
        ref_pos = query_pos + read_start

        while query_pos < len(read_cigar):
            if read_cigar[query_pos] in ["=", "M"]:
                dict_coverage[ref_pos] = dict_coverage[ref_pos] + 1
            if read_cigar[query_pos] in refmoveset:
                ref_pos += 1
            query_pos += 1

        reads_info[read_id] = [read_start+1, ref_pos+1]

    return dict_coverage, reads_info


def GetReferences(content, mapped):
    # get length of references
    ref_info = []
    for line in content:
        if line.rstrip().split('\t')[0][:3] == '@PG':
            break
        if line.rstrip().split('\t')[0][:3] == '@SQ':
            # reference = line.rstrip().split('\t')[1].split(':')[1]
            reference = line.rstrip().split('\t')[1][3:]
            print(reference)
            if reference in mapped:
                length_ref = int(line.rstrip().split('\t')[2].split(':')[1])
                ref_info.append([reference, length_ref])
    
    return ref_info

def LoadData(samfile):
    mapped = defaultdict(list)
    with open(samfile, 'r') as f:
        content = f.readlines()
        for i in range(len(content)):
            if content[i].rstrip().split('\t')[0][:3] not in ['@PG', '@SQ', '@HD'] and content[i].rstrip().split('\t')[5] != '*':
                read_id = content[i].rstrip().split('\t')[0]
                start_pos = int(content[i].rstrip().split('\t')[3])
                aligned_ref = content[i].rstrip().split('\t')[2]
                cigar_string = content[i].rstrip().split('\t')[5]
                mapped[aligned_ref].append([read_id, start_pos, cigar_string])
    print(mapped.keys())
    for k, v in mapped.items():
        print(k, len(v))
    # get references and their length
    ref_info = GetReferences(content[1:], mapped)

    return ref_info, mapped

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samfile', type=str, help='path to SAM file')
    parser.add_argument('--output_dir', type=str, help='path to output directory', default=os.getcwd())
    parser.add_argument('--mapped_reads', help="store id of mapped reads into a tsv file", action='store_true', default=False)
    args = parser.parse_args()
    
    # get references
    ref_info, alignments = LoadData(args.samfile)

    for ref, length_ref in ref_info.items():

        dict_coverage, reads_info = GetCoverage(alignments[ref], length_ref)

        with open(os.path.join(args.output_dir, f'{ref.replace(" ", "-")}-cov-pos.tsv'), 'w') as out_f:
            for k, v in dict_coverage.items():
                out_f.write(f'{k}\t{v}\n')
            
        # compute mean coverage
        mean_cov = round(sum(dict_coverage.values())/length_ref, 3)
        with open(os.path.join(args.output_dir, f'{ref.replace(" ", "-")}-cov-mean.tsv'), 'w') as out_f:
            out_f.write(f'{length_ref}\t{mean_cov}\n')

        if args.mapped_reads:
            with open(os.path.join(args.output_dir, f'{samfile.split("/")[-1].split(".")[0]}_mapped_reads.tsv'), 'w') as outfile:
                for k, v in reads_info.items():
                    outfile.write(f'{k}\t{v[0]}\t{v[1]}\n')


if __name__ == '__main__':
    main()
