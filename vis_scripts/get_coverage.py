import sys
from collections import defaultdict
import math
import multiprocessing as mp

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

def get_data(samfile):
    alignments = defaultdict(list)
    with open(f'reads_info.tsv', 'w') as outfile:
        with open(samfile, 'r') as f:
            content = f.readlines()
            for i in range(len(content)):
                read_id = content[i].rstrip().split("\t")[0]
                start_pos = content[i].rstrip().split("\t")[3]
                if content[i].rstrip().split('\t')[0][:3] not in ['@PG', '@SQ', '@HD'] and content[i].rstrip().split('\t')[5] != '*':
                    alignments[content[i].rstrip().split('\t')[2]].append([int(content[i].rstrip().split('\t')[3]), content[i].rstrip().split('\t')[5]])
                    outfile.write(f'{read_id}\t{start_pos}\n')
    # get references and their length
    ref = get_references(content[1:], alignments)

    return ref, alignments

def main():
    samfile = sys.argv[1]
    nprocs = int(sys.argv[2])
    
    # get references
    ref_info, alignments = get_data(samfile)
    print(ref_info)
    
    # determine the size of each subtask
    size = math.ceil(len(alignments)/nprocs)

    # determine the references in each subtasks
    chunks = []
    data = {}
    for k, v in alignments.items():
        if len(data) < size:
            data.update({k: v})
        else:
            chunks.append(data)
            data = {k: v}
    if len(chunks) < nprocs:
        chunks.append(data)

    num_refs = sum([len(i) for i in chunks])
    print(size, len(alignments), len(ref_info), len(chunks), nprocs, num_refs)

    with mp.Manager() as manager:
        results = manager.dict()
        processes = [mp.Process(target=get_coverage, args=(alignments[ref_info[i][0]], ref_info[i][1], results, i)) for i in range(len(ref_info))]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        for process_id, ref_results in results.items():
            with open(f'{ref_info[process_id][0].replace(" ", "-")}-cov-pos.tsv', 'w') as out_f:
                for k, v in ref_results.items():
                    out_f.write(f'{k}\t{v}\n')
            # compute mean coverage
            mean_cov = round(sum(ref_results.values())/ref_info[process_id][1], 3)
            with open(f'{ref_info[process_id][0].replace(" ", "-")}-cov-mean.tsv', 'w') as out_f:
                out_f.write(f'{k}\t{mean_cov}\n')


if __name__ == '__main__':
    main()
