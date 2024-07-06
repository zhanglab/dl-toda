import os
import argparse
import json
import random
from collections import defaultdict
import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='path to output directory')
    parser.add_argument('--output_dir', type=str, help='path to output directory', default=os.getcwd())
    parser.add_argument('--n_reads', type=int, help='number of reads per taxon')
    parser.add_argument('--label', type=int, help='label associated with taxa')
    parser.add_argument('--output_filename', type=str, help='name of output fastq file')
    args = parser.parse_args()

    fq_files = glob.glob(os.path.join(args.input_dir, '*.fq'))
    random.shuffle(fq_files)
    count_reads = 0
    with open(os.path.join(args.output_dir, args.output_filename), 'w') as out_f:
        for i in range(len(fq_files)):
            with open(fq_files[i], 'r') as handle:
                rec = ''
                n_line = 0
                for line in handle:
                    rec += line
                    n_line += 1
                    if n_line == 4:
                        label = int(rec.split('\n')[0].rstrip().split('|')[1])
                        if label != args.label and count_reads <args.n_reads:
                            out_f.write(rec)
                            count_reads += 1
                        # initialize variables again
                        n_line = 0
                        rec = ''


if __name__ == "__main__":
    main()