import os
import sys
import subprocess

def main():
    input_file = sys.argv[1]
    read_length = sys.argv[2]
    coverage = sys.argv[3]
    output_dir = sys.argv[4]
    path_to_art = sys.argv[5] # '/opt/software/ART/2016.06.05-GCCcore-6.4.0/bin/art_illumina'


    fasta_file = input_file.rstrip().split('\t')[0]
    genome_id = '_'.join(fasta_file.rstrip().split('/')[-1].split('_')[2:4])

    if genome_id != input_file.rstrip().split('\t')[1]:
        sys.exit(f'{genome_id}: problem in input file')

    output_file = os.path.join(output_dir, genome_id)
    label = input_file.rstrip().split('\t')[2]
    prefix_id = f'label|{label}|-'

    fold_coverage = coverage #the fold of read coverage to be simulated or number of reads/read pairs generated for each amplicon
    mean_frag_length = 200 #the mean size of DNA/RNA fragments for paired-end simulations
    sdev_frag_length = 10 #the standard deviation of DNA/RNA fragment size for paired-end simulations
    
    print(f'fasta file: {fasta_file}\nread length: {read_length}\ncoverage: {coverage}\ngenome id: {genome_id}\noutput file: {output_file}\nfold coverage: {fold_coverage}\tmean fragment length: {mean_frag_length}\tstandard deviation fragment length: {sdev_frag_length}')
     
    result = subprocess.run([path_to_art, '-ss', 'MSv1', '-i', f'{fasta_file}', '-d', f'{prefix_id}', '-na', '-s', f'{sdev_frag_length}', '-m', f'{mean_frag_length}', '-l', f'{read_length}', '-f', f'{fold_coverage}', '-p', '-o', f'{output_file}'])

    print(result.stdout)
    print(result.stderr)

if __name__ == "__main__":
    main()
