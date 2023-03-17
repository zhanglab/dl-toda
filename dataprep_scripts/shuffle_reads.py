import os
import sys
import random


if __name__ == "__main__":
    input_fq = sys.argv[1]
    output_dir = sys.argv[2]

    output_file = os.path.join(output_dir, input_fq.split('/')[-1][:-3] + '-shuffled.fq')
    with open(output_file) as out_f:
        with open(input_fq, 'r') as f:
            content = f.readlines()
            reads = [''.join(content[j:j+4]) for j in range(0, len(content), 4)]
            random.shuffle(reads)
            out_f.write(''.join(reads))



