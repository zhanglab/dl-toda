import os
import re
from collections import Counter


def process_fq_file(file_path):
    data_dict = {}
    dna_sequences = []


    with open(file_path, 'r') as f:
        lines = f.readlines()


    #Process each section (4 lines per section)
    for i in range(0, len(lines), 4):
        #Extract key and DNA sequence
        sequence_key = lines[i].strip()
        dna_sequence = lines[i+1].strip()


        #Extract the number between the | symbols (keys)
        match = re.search(r'\|(\d+)\|', sequence_key)
        if match:
            key = match.group(1)
            data_dict[key] = dna_sequence
            dna_sequences.append(dna_sequence)


    #Count frequency of each DNA sequence (values)
    sequence_count = dict(Counter(dna_sequences))


    return data_dict, sequence_count


file_path = '/work/pi_zhuzhang_uri_edu/Kaitlyn_Lum/Python_Script/finetuning_l239_test_data_k4_cleaned.fq'
data, sequence_count = process_fq_file(file_path)


if data:
    print("Processed Data:", data)
    print("DNA Sequence Counts:", sequence_count)
    print("Test:", data))
