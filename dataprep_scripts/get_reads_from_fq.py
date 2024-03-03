import os
import argparse
import random

dl_toda_dir = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])

def get_reads(args, input_fq, target):
    reads = {}
    others = []
    with open(input_fq, 'r') as f:
        rec = ''
        n_line = 0
        for line in f:
            rec += line
            n_line += 1
            if n_line == 4:
                read_id = rec.split('\n')[0].rstrip()
                if (args.datatype == 'label' and read_id.split('|')[1] == target) or (args.datatype == 'sequence_id' and read_id.split('-')[0][1:] == target):
                    reads[read_id] = rec
                else:
                    others.append(rec)
                n_line = 0
                rec = ''
    return reads, others


def get_fw_rv_reads(args, reads):
    fw_out_reads = {}
    rv_out_reads = {}
    for read_id, rec in reads.items():
        if read_id[-1] == '2':
            rv_out_reads[read_id[:-2]] = rec
        elif read_id[-1] == '1':
            fw_out_reads[read_id[:-2]] = rec
    return fw_out_reads, rv_out_reads


def split_reads(fw_reads, rv_reads):
    fw_reads_id = [k for k in fw_reads.keys()]
    rv_reads_id = [k for k in rv_reads.keys()]
    unpaired_reads_id = list(set(fw_reads_id).difference(set(rv_reads_id))) + list(set(rv_reads_id).difference(set(fw_reads_id)))
    paired_reads_id = set(fw_reads_id).intersection(set(rv_reads_id))
    return unpaired_reads_id, list(paired_reads_id)


def create_fq_files(args, reads_id, fw_reads, rv_reads, type_reads, output_file):
    if len(reads_id) != 0:
        if type_reads == 'paired':
            rv_output_file = f'{output_file}-rv-paired.fq'
            fw_output_file = f'{output_file}-fw-paired.fq'
            paired_rv_reads = [rv_reads[i] for i in reads_id]
            paired_fw_reads = [fw_reads[i] for i in reads_id]
            with open(rv_output_file, 'w') as f:
                f.write(''.join(paired_rv_reads))
            with open(fw_output_file, 'w') as f:
                f.write(''.join(paired_fw_reads))
            with open(os.path.join(args.output_dir, 'paired-done.txt'), 'w') as f:
                pass
        elif type_reads == 'unpaired':
            unpaired_reads = []
            for i in range(len(reads_id)):
                if reads_id[i] in rv_reads:
                    unpaired_reads.append(rv_reads[reads_id[i]])
                elif reads_id[i] in fw_reads:
                    unpaired_reads.append(fw_reads[reads_id[i]])
            with open(f'{output_file}-unpaired.fq', 'w') as f:
                f.write(''.join(unpaired_reads))
            with open(os.path.join(args.output_dir, 'unpaired-done.txt'), 'w') as f:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fq', help="input fastq file", required=True)
    parser.add_argument('--output_dir', help="output directory", default=os.getcwd())
    parser.add_argument('--datatype', help="extract reads based on label or sequence id", choices=['sequence_id', 'label'], default='label')
    # parser.add_argument('--input', help="list of sequences id or species", nargs="+", required=True)
    parser.add_argument('--input', type=str, help="sequences id or label", required=True)
    parser.add_argument('--paired', action='store_true', help='differentiate reads based on pairs', default=False)
    parser.add_argument('--others', action='store_true', help='group remaining reads into a fq file', default=False)
    args = parser.parse_args()

    # create output directory
    if not os.path.exists(args.output_dir):
        # replace spaces by underscores
        args.output_dir = args.output_dir.replace(" ", "_") 
        os.makedirs(args.output_dir)

    if args.datatype == 'label':
        # lookup for ncbi species associated with given label and get all labels of identical ncbi species
        with open(os.path.join(args.output_dir, 'list_labels'), 'w') as out_f:
            with open(os.path.join(dl_toda_dir, 'data', 'dl_toda_taxonomy.tsv'), 'r') as in_f:
                content = in_f.readlines()
                taxa = {line.rstrip().split('\t')[0]: line.rstrip().split('\t')[2].split(';')[0] for line in in_f}
            print(taxa)
            ncbi_sp = taxa[args.input]
            inputs = [k for k, v in taxa.items() if v == ncbi_sp]
            for l in inputs:
                out_f.write(f'{l}\n')
                
    for i in range(len(inputs)):
        # define output fastq file
        output_file = os.path.join(args.output_dir, f'{args.input_fq.split("/")[-1][:-3]}-{inputs[i]}.fq')
        reads, others = get_reads(args, args.input_fq, inputs[i])
        if args.paired:
            # get fw and rv reads
            fw_reads, rv_reads = get_fw_rv_reads(args, reads)
            # split reads between paired and unpaired
            unpaired_reads_id, paired_reads_id = split_reads(fw_reads, rv_reads)
            # create output fq files
            create_fq_files(args, unpaired_reads_id, fw_reads, rv_reads, "unpaired", output_file)
            create_fq_files(args, paired_reads_id, fw_reads, rv_reads, "paired", output_file)
        else:
            with open(output_file, 'w') as f:
                f.write(''.join(list(reads.values())))
            if args.others:
                others_output_file = os.path.join(args.output_dir, f'{args.input_fq.split("/")[-1][:-3]}-others.fq')
                with open(others_output_file, 'w') as f:
                    f.write(''.join(others))
