import sys
from collections import defaultdict
import math
import multiprocessing as mp
# from mpi4py import MPI

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# nprocs = comm.Get_size()

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

def get_coverage(list_of_reads, length_ref):
    # for each reference in the dictionary, create a new dictionary with the
    # number of matches at each position encountered
    dict_coverage = defaultdict(lambda : 0)

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

    # compute mean coverage
    mean_cov = round(sum(dict_coverage.values())/length_ref, 3)

    return mean_cov

def get_references(content, alignments):
    ref = {}
    for line in content:
        if line.rstrip().split('\t')[0][:3] == '@SQ' and line.rstrip().split('\t')[1].split(':')[1] in alignments:
            ref[line.rstrip().split('\t')[1].split(':')[1]] = int(line.rstrip().split('\t')[2].split(':')[1])
    return ref

def get_data(samfile):
    alignments = defaultdict(list)
    with open(samfile, 'r') as f:
        content = f.readlines()
        for i in range(len(content)):
            if content[i].rstrip().split('\t')[0][:3] not in ['@PG', '@SQ', '@HD'] and content[i].rstrip().split('\t')[5] != '*':
                alignments[content[i].rstrip().split('\t')[2]].append([int(content[i].rstrip().split('\t')[3]), content[i].rstrip().split('\t')[5]])
    # get references and their length
    ref = get_references(content[1:], alignments)

    return ref, alignments

def main():
    samfile = sys.argv[1]
    nprocs = int(sys.argv[2])
    # if rank == 0:
    # get references
    ref_info, alignments = get_data(samfile)
    print(ref_info)
    # determine the size of each subtask
    size = math.ceil(len(alignments)/nprocs)
    # determine the references in each subtasks
    # chunk_size = math.ceil(len(fq_files)/nprocs)
    # grouped_files = [fq_files[i:i+chunk_size] for i in range(0, len(fq_files), chunk_size)]

    chunks = []
    data = {}
    for k, v in alignments.items():
        print(f'{k}\t{len(v)}\n')
        if len(data) < size:
            data.update({k: v})
        else:
            chunks.append(data)
            data = {k: v}
    if len(chunks) < nprocs:
        chunks.append(data)

    num_refs = sum([len(i) for i in chunks])
    print(size, len(alignments), len(ref_info), len(chunks), nprocs, num_refs)
    for i in chunks:
        print(len(i))

    # with mp.Manager() as manager:
    #     train_reads = manager.dict()
    #     val_reads = manager.dict()
    #     if args.dataset_type == 'sim':
    #         processes = [mp.Process(target=create_tfrecords, args=(args, grouped_files[i])) for i in range(len(grouped_files))]
    #     elif args.dataset_type == 'meta':
    #         processes = [mp.Process(target=create_meta_tfrecords, args=(args, grouped_files[i])) for i in range(len(grouped_files))]
    #     for p in processes:
    #         p.start()
    #     for p in processes:
    #         p.join()

    # else:
    #     chunks = None
    #     ref_info = None
    #     alignments = None
    
    # distribute the subtasks to the processes
    # try:
    #     chunks = comm.scatter(chunks, root=0)
    # except ValueError:
    #     MPI.COMM_WORLD.Abort(1)

    # broadcast info about references to all processes
    # ref_info = comm.bcast(ref_info, root=0)
    # # get coverage for each reference
    # with open(f'process-{rank}-cov.tsv', 'w') as out_f:
    #    for ref, reads in chunks.items():
    #         if rank == 1:
    #             print(f'{ref}\t{len(reads)}\t{ref_info[ref]}')
    #         mean_cov = get_coverage(reads, ref_info[ref])
    #         out_f.write(f'{ref}\t{ref_info[ref]}\t{len(reads)}\t{mean_cov}\n')
    # print(f'process {rank} received {len(chunks)} references and the reference info dictionary with {len(ref_info)} entries')


if __name__ == '__main__':
    main()
