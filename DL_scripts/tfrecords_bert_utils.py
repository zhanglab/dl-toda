import numpy as np
import random


def load_dnabert_seq(args):
    with open(args.input, 'r') as f:
        content = f.readlines()
        labels = [s.rstrip().split('\t')[0] for s in content]
        dna_sequences = [s.rstrip().split('\t')[1].split(" ") for s in content]
        # random.shuffle(dna_sequences)

    return dna_sequences, labels

# def split_read(args, reads, read, r_index, process):
#     # randomly choose whether to have segment 2 not coming after segment 1 (True) or keeping the read unchanged (False)
#     nsp_choice = random.choice([True, False])
#     # define first segment
#     segment_1 = read[:len(read)//2]
#     if nsp_choice == False or args.bert_step == 'finetuning':
#         print('nsp not implemented')
#         nsp_label = 1 # 'IsNext' --> verified with bert code on sample text
#         segment_2 = read[len(read)//2:]
#     else:
#         nsp_label = 0 # 'NotNext' --> verified with bert code on sample text
#         # randomly select another sequence in the pool of sequences
#         o_index = random.choice([i for i in range(len(reads)) if i!= r_index])
#         # split selected sequence in two segments of equal length
#         o_seq = reads[o_index].rstrip().split('\n')[1]
#         # randomly select one segment
#         segment_2 = random.choice([o_seq[:len(o_seq)//2], o_seq[len(o_seq)//2:]])
    
#     return segment_1, segment_2, nsp_label

def split_read(args, reads, read, r_index):
    # randomly choose whether to have segment 2 not coming after segment 1 (True) or keeping the read unchanged (False)
    #nsp_choice = random.choice([True, False])
    nsp_choice = False
    # define first segment
    segment_1 = read[:len(read)//2]
    if nsp_choice == False or args.bert_step == 'finetuning':
        # print('nsp not implemented')
        nsp_label = 1 # 'IsNext' --> verified with bert code on sample text
        segment_2 = read[len(read)//2:]
    else:
        nsp_label = 0 # 'NotNext' --> verified with bert code on sample text
        # randomly select another sequence in the pool of sequences
        o_index = random.choice([i for i in range(len(reads)) if i!= r_index])
        # split selected sequence in two segments of equal length
        o_seq = reads[o_index].rstrip().split('\n')[1]
        # randomly select one segment
        segment_2 = random.choice([o_seq[:len(o_seq)//2], o_seq[len(o_seq)//2:]])
    
    return segment_1, segment_2, nsp_label

    

def get_nsp_input(args, segment_1, segment_2):
# def get_nsp_input(args, bases_list):
    # create 2 segments from list
    # segment_1 = bases_list[:len(bases_list)//2]
    # segment_2 = bases_list[len(bases_list)//2:]

    # randomly choose whether to have segment 2 after segment 1 or not 
    # nsp_choice = random.choice([True, False])
    # generate random segment 2 in case nsp_choice is False
    # if nsp_choice:
    #     nsp_label = 0 # 'NotNext'
    #     kmers = [k for k in args.dict_kmers.keys() if k not in ["[UNK]", "[MASK]", "[CLS]", "[SEP]", "[PAD]"]]
    #     segment_2 = [args.dict_kmers[x] for x in random.choices(kmers, k=len(segment_2))]
    # else:
    #     nsp_label = 1 # 'IsNext'

    ## concatenate segments and add CLS and SEP tokens
    #concatenate_segments = [args.dict_kmers['[CLS]']] + segment_1 + [args.dict_kmers['[SEP]']] + segment_2 + [args.dict_kmers['[SEP]']]
    # NSP is not implemented yet
    concatenate_segments = [args.dict_kmers['[CLS]']] + segment_1 + segment_2 + [args.dict_kmers['[SEP]']]
    ## create list of segment ids
    #segment_ids = [0]*(2+len(segment_1)) + [1]*(1+len(segment_2))
    segment_ids = [0]*(1+len(segment_1)) + [0]*(len(segment_2)+1)

    return np.array(concatenate_segments), np.array(segment_ids)


def get_masked_array(args, mask_lm_positions, input_array):
    output = input_array.copy()
    # replace each chosen base by the MASK token number 80% of the time, a random base 10% of the time
    # or the unchanged base 10% of the time
    replacements = ["masked", "random", "same"]
    weights = [0.8, 0.1, 0.1]
    for i in range(len(output)):
        if i in mask_lm_positions:
            # randomly choose one type of replacement
            r_type = random.choices(replacements, weights=weights)
            if r_type[0] == 'masked':
                output[i] = args.dict_kmers["[MASK]"]
            elif r_type[0] == 'random':
                output[i] = random.choices([args.dict_kmers[k] for k in args.dict_kmers.keys() if k not in ["[UNK]", "[MASK]", "[CLS]", "[SEP]", "[PAD]"]])[0]
            elif r_type[0] == 'same':
                continue

    return output

def get_mlm_input(args, input_array):
    # compute number of bases to mask (take into account 2*'SEP' and 'CLS')
    # n_mask = int(args.masked_lm_prob * (len(input_array)-3)) # --> could be updated to mask predictions per sequence
    # compute number of bases to mask (take into account 'CLS') --> without nsp
    n_mask = int(args.masked_lm_prob * (len(input_array)-1))
    # if args.contiguous:
    #     # mask contiguous bases
    #     # choose index of first base to mask
    #     start_mask_idx = random.choice(range(0, args.kmer_vector_length - n_mask))
    #     mask_indexes = list(range(start_mask_idx,start_mask_idx+n_mask))
    #     # select bases to mask
    #     bases_masked = [False if i not in range(start_mask_idx,start_mask_idx+n_mask) else True for i in range(args.kmer_vector_length)]
    # else:
    # get indexes of SEP, CLS and UNK tokens
    # sep_indices = [i for i in range(len(input_array)) if input_array[i] in [args.dict_kmers['[SEP]'],args.dict_kmers['[CLS]'],args.dict_kmers['[UNK]']]]
    # get indexes of CLS and UNK tokens ()without NSP task
    sep_indices = [i for i in range(len(input_array)) if input_array[i] in [args.dict_kmers['[CLS]'],args.dict_kmers['[UNK]'],args.dict_kmers['[SEP]']]]
    # get list of indices of tokens to mask
    mask_lm_positions = random.sample(list(set(range(len(input_array))) - set(sep_indices)), n_mask)
    # mask bases
    input_ids = get_masked_array(args, mask_lm_positions, input_array)
    # Prepare sample_weights parameter to loss function. The weights have a value of 1.0 for eve
    masked_lm_weights = [1.0] * n_mask # only compute loss for masked k-mers
    # create input_mask vector indicating padded values
    input_mask = [1] * len(input_ids)

    return input_ids, input_mask, masked_lm_weights, mask_lm_positions, [input_array[i] for i in mask_lm_positions]
