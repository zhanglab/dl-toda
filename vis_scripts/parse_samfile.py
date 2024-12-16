import sys
import os
from collections import defaultdict
import math
import argparse
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
from dataprep_scripts.utils import load_fq_file

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


def GetCoverageOfRead(read_start, read_cigar, dict_coverage):
    query_pos = 0
    ref_pos = query_pos + read_start

    while query_pos < len(read_cigar):
        if read_cigar[query_pos] in ["=", "M"]:
            dict_coverage[ref_pos] = dict_coverage[ref_pos] + 1
        if read_cigar[query_pos] in refmoveset:
            ref_pos += 1
        query_pos += 1
    
    return ref_pos


def GetCoverageOfSample(list_of_reads, length_ref, label=None):
    dict_coverage = {i: 0 for i in range(length_ref)}
    reads_info = defaultdict(list)

    for j in range(0, len(list_of_reads)):
        read_id = list_of_reads[j][0]
        read_label = list_of_reads[j][0].split('|')[1]
        read_start = list_of_reads[j][1] - 1
        read_cigar = ExtendCigar(list_of_reads[j][2])

        if label:
            if read_label == label:
                ref_pos = GetCoverageOfRead(read_start, read_cigar, dict_coverage)
                reads_info[read_id] = [read_start+1, ref_pos+1, list_of_reads[j][2]]
        else:
            ref_pos = GetCoverageOfRead(read_start, read_cigar, dict_coverage)
            reads_info[read_id] = [read_start+1, ref_pos+1, list_of_reads[j][2]]

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

    # get references and their length
    ref_info = GetReferences(content[1:], mapped)

    return ref_info, mapped

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samfile', type=str, help='path to SAM file')
    parser.add_argument('--output_dir', type=str, help='path to output directory', default=os.getcwd())
    parser.add_argument('--mapped_reads', help="store id of mapped reads into a tsv file", action='store_true', default=False)
    parser.add_argument('--unmapped_reads', help="store id of unmapped reads into a tsv file", action='store_true', default=False)
    parser.add_argument('--label', type=str, help="label of reads of interest")
    parser.add_argument('--fqfile', type=str, help="path to fastq file")
    args = parser.parse_args()
    
    # get references
    ref_info, alignments = LoadData(args.samfile)

    for i in range(len(ref_info)):
        ref = ref_info[i][0]
        length_ref = ref_info[i][1]

        if args.label:
            dict_coverage, reads_info = GetCoverageOfSample(alignments[ref], length_ref, label=args.label)
        else:
            dict_coverage, reads_info = GetCoverageOfSample(alignments[ref], length_ref)

        with open(os.path.join(args.output_dir, f'{ref.replace(" ", "-")}-cov-pos.tsv'), 'w') as out_f:
            for k, v in dict_coverage.items():
                out_f.write(f'{k}\t{v}\n')
            
        # compute mean coverage
        mean_cov = round(sum(dict_coverage.values())/length_ref, 3)
        with open(os.path.join(args.output_dir, f'{ref.replace(" ", "-")}-cov-mean.tsv'), 'w') as out_f:
            out_f.write(f'number of positions covered\t{sum(dict_coverage.values())}\nreference length\t{length_ref}\naverage coverage\t{mean_cov}\n')

        if args.mapped_reads:
            with open(os.path.join(args.output_dir, f'{args.samfile.split("/")[-1].split(".")[0]}_mapped_reads.tsv'), 'w') as outfile:
                for k, v in reads_info.items():
                    outfile.write(f'{k}\t{v[0]}\t{v[1]}\t{v[2]}\n')

        if args.unmapped_reads:
            reads = load_fq_file(args.fqfile, 4)
            dict_reads_length = {}
            for r in reads:
                read_id = r.split("\n")[0][1:]
                length = len(r.split("\n")[1])
                if r.split("\n")[0].split('|')[1] == args.label:
                    dict_reads_length[read_id] = length

            # get unmapped reads and their length
            unmapped = set(list(dict_reads_length.keys())).difference(set(list(reads_info.keys())))
            with open(os.path.join(args.output_dir, f'unmapped_reads_{args.label}.tsv'), 'w') as f:
                for r in unmapped:
                    f.write(f'{r}\t{dict_reads_length[r]}\n')


if __name__ == '__main__':
    main()
