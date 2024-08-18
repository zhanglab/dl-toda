import os
import sys
import math
import multiprocessing as mp
# sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
# from DL_scripts.tfrecords_utils.py import get_reverse_seq


def get_reverse_seq(read):
    """ Converts a k-mer to its reverse complement """
    translation_dict = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N",
                        "K": "N", "M": "N", "R": "N", "Y": "N", "S": "N",
                        "W": "N", "B": "N", "V": "N", "H": "N", "D": "N",
                        "X": "N"}
    list_bases = list(read)
    # get negative strand, example: 'AGTAGATGATAGGGA' becomes 'TCATCTACTATCCCT'
    list_bases = [translation_dict[base] for base in list_bases]
    # return reverse complement: 'TCCCTATCATCTACT'
    return ''.join(list_bases)[::-1]


def get_kmers(k, y=''):
    bases = ['A', 'T', 'C', 'G']
    if k == 0:
        yield y
    else:
        for b in bases:
            yield from get_kmers(k-1, b+y)

# def get_canonical_kmers(kmers):
#     can_kmers = []
#     while len(kmers) != 0:
#         kmer = kmers.pop(0)
#         # get reverse complement
#         rev_seq = get_reverse_seq(kmer)
#         # select canonical k-mer (k-mer that appears)
#         if kmer <= rev_seq: # y comes before in alphabetical order
#             print(f'{kmer} <= {rev_seq}')
#             can_kmers.append(kmer)
#         else:
#             print(f'{kmer} > {rev_seq}')
#             can_kmers.append(rev_seq)
#         if rev_seq in kmers:
#             # remove reverse complement from list
#             kmers.remove(rev_seq)

#     return can_kmers


def get_canonical_kmers(can_kmers, process_id, list_kmers):
    process_can_kmers = []
    for kmer in list_kmers:
        # get reverse complement
        rev_seq = get_reverse_seq(kmer)
        if kmer <= rev_seq: # y comes before in alphabetical order
            print(f'kmer: {kmer} <= rev_seq: {rev_seq}')
            process_can_kmers.append(kmer)
    can_kmers[str(process_id)] = process_can_kmers



if __name__ == "__main__":
    k_value = int(sys.argv[1])
    num_proc = int(sys.argv[2])
    model = str(sys.argv[3])

    # get list of all k-mers
    kmers = [x for x in get_kmers(k_value)]
    print(len(kmers))

    # compute number of canonical k-mers
    if (k_value % 2) == 0:
        n_kmers = (4 ** k_value + 4 ** (k_value / 2)) / 2
    else:
        n_kmers = (4 ** k_value) / 2

    print(n_kmers)


    # get canonical k-mers
    # can_kmers = get_canonical_kmers(kmers)

    chunk_size = math.ceil(len(kmers)/num_proc)
    grouped_kmers = [kmers[i:i+chunk_size] for i in range(0, len(kmers), chunk_size)]
    print(f'{chunk_size}\t{len(grouped_kmers)}')
    with mp.Manager() as manager:
        can_kmers = manager.dict()
        processes = [mp.Process(target=get_canonical_kmers, args=(can_kmers, i, grouped_kmers[i])) for i in range(len(grouped_kmers))]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        # join all canonical kmers into a list
        list_can_kmers = []
        for process in range(len(can_kmers)):
            print(can_kmers[str(process)])
            list_can_kmers += can_kmers[str(process)]
        print(len(list_can_kmers))
        # get unique canonical kmers
        list_can_kmers = list(set(list_can_kmers)).sort()
        print(len(list_can_kmers))
        with open(f'{k_value}mers.txt', 'w') as f:
            if model == 'BERT':
                f.write('[PAD]\n[CLS]\n[SEP]\n[MASK]\n[UNK]\n')
            else:
                f.write('pad\nunknown\n')
            for k in list(list_can_kmers):
                f.write(f'{k}\n')





