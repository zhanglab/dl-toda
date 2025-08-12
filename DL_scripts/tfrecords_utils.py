import random


def prepare_input_data(read_seq, k_value, step, read_length, dict_kmers, rec=None, read_id=None, pair=False, DNA_model=False, insert_size=False, dataset_type=None):
    bases = {'A': 2, 'T': 3, 'C': 4, 'G': 5}
    # if pair:
    #     # concatenate forward and reverse reads into one vector
    #     if DNA_model:
    #         fw_dna = [bases[x] if x in bases else 1 for x in rec.split('\n')[1].rstrip()]
    #         rv_dna = [bases[x] if x in bases else 1 for x in rec.split('\n')[5].rstrip()]
    #         # update read length to match the max read length
    #         if len(fw_dna) < read_length:
    #             # pad list of bases with 0s to the right
    #             fw_dna = fw_dna + [0] * (read_length - len(fw_dna))
    #             rv_dna = rv_dna + [0] * (read_length - len(rv_dna))
    #     else:
    #         fw_dna, _ = get_kmer_arr(rec.split('\n')[1].rstrip(), k_value, step, read_length, dict_kmers)
    #         rv_dna, _ = get_kmer_arr(rec.split('\n')[5].rstrip(), k_value, step, read_length, dict_kmers)
    #     # combine fw nad rv data into one list
    #     dna_list = fw_dna + rv_dna
    #     # append insert size for dna lists as pairs of reads
    #     if insert_size:
    #         dna_list.append(int(dict_kmers[read_id.split('|')[3]]))
    # else:
        # if DNA_model:
        #     dna_list = [bases[x] if x in bases else 1 for x in read_seq]
        #     # update read length to match the max read length
        #     if len(dna_list) < read_length:
        #         # pad list of bases with 0s to the right
        #         dna_list = dna_list + [0] * (read_length - len(dna_list))
        # else:
    dna_list = get_kmer_arr(read_seq, k_value, step, read_length, dict_kmers, dataset_type='sim')

    return dna_list


def shuffle_reads(fastq_file, num_lines):
    with open(fq_file) as handle:
        content = handle.readlines()
        reads = [''.join(content[j:j + num_lines]) for j in range(0, len(content), num_lines)]
        random.shuffle(reads)

    return reads


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
        for count, line in enumerate(infile, 0):
            kmer = line.rstrip()
            kmer_to_id[kmer] = count
    return kmer_to_id

def get_kmer_index(args, kmer, dict_kmers):
    """Convert kmers into their corresponding index"""
    if kmer in dict_kmers:
        idx = dict_kmers[kmer]
    else:
        if args.canonical_kmers:
            if get_reverse_seq(kmer) in dict_kmers:
                idx = dict_kmers[get_reverse_seq(kmer)]
        else:
            idx = dict_kmers['[UNK]']
    return idx

def get_flipped_reads(args, records):
    flipped_records = []
    for rec in records:
        flipped_read_id = rec.split('\n')[0]
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


def get_kmer_arr(read, k_value, step, max_read_length, dict_kmers, dataset_type=None):
    """ Converts a DNA sequence split into a list of k-mers """

    # adjust read length if above args.read_length
    if dataset_type == 'sim':
        if len(read) > max_read_length:
            read = read[:max_read_length]

    list_kmers = []
    for i in range(0, len(read)-k_value+1, step):
        kmer = read[i:i + k_value]
        idx = get_kmer_index(args, kmer, dict_kmers)
        list_kmers.append(idx)

    return list_kmers
