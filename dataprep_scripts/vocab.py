import os
import sys
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
from DL_scripts.tfrecords_utils.py import get_reverse_seq


def get_kmers(k, y=''):
    bases = ['A', 'T', 'C', 'G']
    if k == 0:
        yield y
    else:
        for b in bases:
            yield from get_kmers(k-1, b+y)


def get_canonical_kmers(kmers):
    can_kmers = []
    while len(kmers) != 0:
        kmer = kmers.pop(0)
        # get reverse complement
        rev_seq = get_reverse_seq(kmer)
        # select canonical k-mer (k-mer that appears)
        if kmer <= rev_seq: # y comes before in alphabetical order
            can_kmers.append(kmer)
        else:
            can_kmers.append(rev_seq)
        # remove reverse complement from list
        kmers.remove(rev_seq)

    return can_kmers


if __name__ == "__main__":
    k_value = int(sys.argv[1])

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
    can_kmers = get_canonical_kmers(kmers)
    print(len(can_kmers))

    with open(f'{k_value}mers.txt', 'w') as f:
        f.write('unknown\n')
        for k in can_kmers:
            f.write(f'{k}\n')





