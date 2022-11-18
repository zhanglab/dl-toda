import numpy as np

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


def vocab_dict(filename):
    """ Returns dictionary mapping kmers to their id. """
    """ Starts index at 1 instead to use 0 as a special padding value.  """
    kmer_to_id = {}
    with open(filename) as infile:
        for count, line in enumerate(infile, 1):
            kmer = line.rstrip()
            kmer_to_id[kmer] = count
    return kmer_to_id

def get_kmer_index(kmer, dict_kmers):
    """Convert kmers into their corresponding index"""
    if kmer in dict_kmers:
        idx = dict_kmers[kmer]
    elif get_reverse_seq(kmer) in dict_kmers:
        idx = dict_kmers[get_reverse_seq(kmer)]
    else:
        idx = dict_kmers['unknown']

    return idx

def get_flipped_reads(args, records):
    flipped_records = []
    for rec in records:
        flipped_read_id = rec.split("\n")[0]
        flipped_read = rec.split('\n')[1][::-1]
        flipped_qual = rec.split('\n')[3][::-1]
        flipped_records.append(f'{flipped_read_id}-f\n{flipped_read}\n+\n{flipped_qual}')
    print(f'# flipped reads: {len(flipped_records)}\t{len(records)}')
    with open((args.input_fastq[:-3] + '-flipped.fq'), 'w') as out_f:
        out_f.write('\n'.join(flipped_records))

    return flipped_records


def cut_read(args, read):
    list_reads = []
    for i in range(0, len(read), args.read_length):
        list_reads.append(read[i:i+args.read_length])
    return list_reads


def get_kmer_arr(args, read):
    """ Converts a DNA sequence split into a list of k-mers """
    list_kmers = []
    for i in range(0, len(read)-args.k_value+1, 1):
        kmer = read[i:i + args.k_value]
        idx = get_kmer_index(kmer, args.dict_kmers)
        list_kmers.append(idx)

    if len(list_kmers) < args.kmer_vector_length:
        # pad list of kmers with 0s to the right
        list_kmers = list_kmers + [0] * (args.kmer_vector_length - len(list_kmers))
    return np.array(list_kmers)
