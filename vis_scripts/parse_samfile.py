import sys
import os
from collections import defaultdict
import math
import multiprocessing as mp
import argparse

def extend_cigar(cigar):
    new_cigar = ''
    num = ''
    for i in cigar:
        if i.isnumeric():
            num += i
        else:
            new_cigar += i*int(num)
            num = ''
    return new_cigar

def get_coverage(list_of_reads, length_ref, results, process_id):
    # for each reference in the dictionary, create a new dictionary with the
    # number of matches at each position encountered
    # dict_coverage = defaultdict(lambda : 0)
    dict_coverage = {i: 0 for i in range(length_ref)}

    for j in range(0, len(list_of_reads)):
        read_start = list_of_reads[j][0] - 1 # read_start = list_of_reads[j][0] - 1
        read_cigar = extend_cigar(list_of_reads[j][1]) # read_cigar = list_of_reads[j][1]
        refmoveset = {'M', '=', 'X', 'D', 'N'}
        # refnomoveset = {'I', 'S', 'H', 'P'}
        query_pos = 0
        ref_pos = query_pos + read_start

        while query_pos < len(read_cigar):
            if read_cigar[query_pos] in ["=", "M"]:
                dict_coverage[ref_pos] = dict_coverage[ref_pos] + 1
            if read_cigar[query_pos] in refmoveset:
                ref_pos += 1
            query_pos += 1

    results[process_id] = dict_coverage


def get_references(content, alignments):
    ref = {}
    for i, line in enumerate(content):
    # for line in content:
        if line.rstrip().split('\t')[0][:3] == '@SQ' and line.rstrip().split('\t')[1].split(':')[1] in alignments:
            ref[i] = [line.rstrip().split('\t')[1].split(':')[1], int(line.rstrip().split('\t')[2].split(':')[1])]
            # ref[line.rstrip().split('\t')[1].split(':')[1]] = int(line.rstrip().split('\t')[2].split(':')[1])
    return ref

def get_data(args, samfile):
    alignments = defaultdict(list)
    with open(os.path.join(args.output_dir, f'{samfile.split("/")[-1].split(".")[0]}_mapped_reads.tsv'), 'w') as outfile:
        with open(samfile, 'r') as f:
            content = f.readlines()
            for i in range(len(content)):
                if content[i].rstrip().split('\t')[0][:3] not in ['@PG', '@SQ', '@HD'] and content[i].rstrip().split('\t')[5] != '*':
                    alignments[content[i].rstrip().split('\t')[2]].append([int(content[i].rstrip().split('\t')[3]), content[i].rstrip().split('\t')[5]])
                    read_id = content[i].rstrip().split("\t")[0]
                    start_pos = content[i].rstrip().split("\t")[3]
                    outfile.write(f'{read_id}\t{start_pos}\n')
    # get references and their length
    ref = get_references(content[1:], alignments)

    return ref, alignments

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samfile', type=str, help='path to SAM file')
    parser.add_argument('--output_dir', type=str, help='path to output directory', default=os.getcwd())
    parser.add_argument('--num_processes', type=int, default=8)
    parser.add_argument('--coverage', help="compute coverage of reference sequences", action='store_true', default=False)
    args = parser.parse_args()
    
    # get references
    ref_info, alignments = get_data(args.samfile)
    print(ref_info)

    if args.coverage:
        # determine the size of each subtask
        size = math.ceil(len(alignments)/args.num_processes)

        # determine the references in each subtasks
        chunks = []
        data = {}
        for k, v in alignments.items():
            if len(data) < size:
                data.update({k: v})
            else:
                chunks.append(data)
                data = {k: v}
        if len(chunks) < args.num_processes:
            chunks.append(data)

        num_refs = sum([len(i) for i in chunks])
        print(size, len(alignments), len(ref_info), len(chunks), args.num_processes, num_refs)

        with mp.Manager() as manager:
            results = manager.dict()
            processes = [mp.Process(target=get_coverage, args=(alignments[ref_info[i][0]], ref_info[i][1], results, i)) for i in range(len(ref_info))]
            for p in processes:
                p.start()
            for p in processes:
                p.join()

            for process_id, ref_results in results.items():
                with open(os.path.join(args.output_dir, f'{ref_info[process_id][0].replace(" ", "-")}-cov-pos.tsv'), 'w') as out_f:
                    for k, v in ref_results.items():
                        out_f.write(f'{k}\t{v}\n')
                # compute mean coverage
                mean_cov = round(sum(ref_results.values())/ref_info[process_id][1], 3)
                with open(os.path.join(args.output_dir, f'{ref_info[process_id][0].replace(" ", "-")}-cov-mean.tsv'), 'w') as out_f:
                    out_f.write(f'{k}\t{mean_cov}\n')


if __name__ == '__main__':
    main()
