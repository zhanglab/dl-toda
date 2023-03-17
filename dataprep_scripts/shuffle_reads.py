import os
import sys
import random
from Bio import SeqIO


if __name__ == "__main__":
    input_fq = sys.argv[1]
    output_dir = sys.argv[2]

    output_file = os.path.join(output_dir, input_fq.split('/')[-1][:-3] + '-shuffled.fq')
    with open(output_file, 'w') as out_f:
        reads = list(SeqIO.parse(input_fq, "fastq"))
        random.shuffle(reads)
        out_f.write(''.join(reads))



