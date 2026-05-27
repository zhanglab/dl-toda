import os
import sys
import random


if __name__ == "__main__":
    input_fq = sys.argv[1]
    output_dir = sys.argv[2]

    output_file = os.path.join(output_dir, input_fq.split('/')[-1][:-3] + '-shuffled.fq')
    with open(output_file, 'w') as output_handle:
        with open(input_fq) as input_handle:
            content = input_handle.readlines()
            reads = [''.join(content[j:j + 4]) for j in range(0, len(content), 4)]
            random.shuffle(reads)
        output_handle.write(''.join(reads))


