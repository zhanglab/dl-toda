import os
import argparse


def get_reads(input_fq, target):
    fw_out_reads = {}
    rv_out_reads = {}
    with open(input_fq, 'r') as f:
        rec = ''
        n_line = 0
        for line in f:
            rec += line
            n_line += 1
            if n_line == 4:
                read_id = rec.split('\n')[0].rstrip()
                if target in read_id:
                    if read_id[-1] == '2':
                        rv_out_reads[read_id[:-2]] = rec
                    elif read_id[-1] == '1':
                        fw_out_reads[read_id[:-2]] = rec
                n_line = 0
                rec = ''
    return fw_out_reads, rv_out_reads


def split_reads(fw_reads, rv_reads):
    fw_reads_id = [k[:-2] for k in fw_reads.keys()]
    rv_reads_id = [k[:-2] for k in rv_reads.keys()]
    unpaired_reads_id = list(set(fw_reads_id).difference(set(rv_reads_id))) + list(set(rv_reads_id).difference(set(fw_reads_id)))
    paired_reads_id = set(fw_reads_id).intersection(set(rv_reads_id))
    return unpaired_reads_id, list(paired_reads_id)


def create_fq_files(reads_id, fw_reads, rv_reads, type_reads, output_file):
    if len(reads_id) != 0:
        if type_reads == 'paired':
            rv_output_file = f'{output_file}-paired-rv.fq'
            fw_output_file = f'{output_file}-paired-fw.fq'
            paired_rv_reads = [rv_reads[f'{i}/2'] for i in reads_id]
            paired_fw_reads = [fw_reads[f'{i}/1'] for i in reads_id]
            with open(rv_output_file, 'w') as f:
                f.write(''.join(paired_rv_reads))
            with open(fw_output_file, 'w') as f:
                f.write(''.join(paired_fw_reads))
        elif type_reads == 'unpaired':
            unpaired_reads = []
            for i in range(len(reads_id)):
                if f'{reads_id[i]}/2' in rv_reads:
                    unpaired_reads.append(rv_reads[f'{reads_id[i]}/2'])
                elif f'{reads_id[i]}/1' in fw_reads:
                    unpaired_reads.append(fw_reads[f'{reads_id[i]}/1'])
            with open(f'{output_file}-unpaired.fq', 'w') as f:
                f.write(''.join(unpaired_reads))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fq', help="input fastq file", required=True)
    parser.add_argument('--output_dir', help="output directory", default=os.getcwd())
    parser.add_argument('--input', help="list of input labels or sequences id", nargs="+", required=True)
    args = parser.parse_args()

    for i in range(len(args.input)):
        # define output fastq file
        output_file = os.path.join(args.output_dir, f'{args.input_fq.split("/")[-1][:-6]}-{args.input[i]}')
        # load fw and rv reads
        fw_reads, rv_reads = get_reads(args.input_fq, args.input[i])
        # split reads between paired and unpaired
        unpaired_reads_id, paired_reads_id = split_reads(fw_reads, rv_reads)
        # create output fq files
        create_fq_files(unpaired_reads_id, fw_reads, rv_reads, "unpaired", output_file)
        create_fq_files(paired_reads_id, fw_reads, rv_reads, "paired", output_file)
