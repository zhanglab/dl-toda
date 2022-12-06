import os
import sys
import glob
import subprocess

def main():
    input_data = sys.argv[1]
    read_length = sys.argv[2]
    coverage = sys.argv[3]
    output_dir = sys.argv[4]
    input_dir = sys.argv[5] # path to directory containing fasta files
    path_to_art = sys.argv[6]

    genome_id = input_data.rstrip().split('\t')[0]
    fasta_file = glob.glob(os.path.join(input_dir, f'*{genome_id}*.fna'))

    output_file = os.path.join(output_dir, genome_id)
    label = input_data.rstrip().split('\t')[1]
    prefix_id = f'label|{label}|-'

    fold_coverage = coverage #the fold of read coverage to be simulated or number of reads/read pairs generated for each amplicon

    print(f'fasta file: {fasta_file}\nread length: {read_length}\ncoverage: {coverage}\ngenome id: {genome_id}\noutput file: {output_file}\nfold coverage: {fold_coverage}')

    result = subprocess.run([path_to_art, '-ss', 'MSv1', '-i', f'{fasta_file}', '-d', f'{prefix_id}', '-na', '-l', f'{read_length}', '-f', f'{fold_coverage}', '-p', '-o', f'{output_file}'])

if __name__ == "__main__":
    main()
