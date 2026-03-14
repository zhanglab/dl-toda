import glob
import sys
import random
import multiprocessing as mp
import os
import math
import argparse
import pandas as pd
import statistics
from Bio import SeqIO
import subprocess
import zipfile
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
dnabert_exec = '/work/pi_yingzhang_uri_edu/ccres/tools/DNABERT/examples/data_process_template/'
sys.path.append(dnabert_exec)
from process_pretrain_data import cut_no_overlap, sampling, get_kmer_sentence
from select_genomes import get_gtdb_info

MIN_IDENTITY = 70
TICKS_INTERVAL = 500000
blastn_exec = "/modules/uri_apps/software/BLAST+/2.15.0-gompi-2023a/bin/blastn"
makeblastdb_exec = "/modules/uri_apps/software/BLAST+/2.15.0-gompi-2023a/bin/makeblastdb"
ncbi_datasets_exec = "/work/pi_yingzhang_uri_edu/ccres/tools/datasets"

def SelectGenomes(args):
    # load gtdb info
    genomes, _, ncbi_genome_category, ncbi_genome_representation, gtdb_rep_genome, gtdb_taxonomy = get_gtdb_info(args.gtdb_info)
    idx_pos_species = [i for i in range(len(gtdb_taxonomy)) if gtdb_taxonomy[i].split(';')[-1].split('__')[1] == args.pos_species]
    idx_neg_species = [i for i in range(len(gtdb_taxonomy)) if gtdb_taxonomy[i].split(';')[-1].split('__')[1] == args.neg_species]
    pos_train_genomes = []
    pos_test_genomes = []
    for idx in idx_pos_species:
        if ncbi_genome_representation[idx] == 'full':
            if ncbi_genome_category[idx] == 'derived from single cell':
                pos_train_genomes.append(idx)
            elif ncbi_genome_category[idx] == 'derived from metagenome':
                pos_test_genomes.append(idx)
    neg_train_genomes = []
    neg_test_genomes = []
    for idx in idx_neg_species:
        if ncbi_genome_representation[idx] == 'full':
            if ncbi_genome_category[idx] == 'derived from single cell':
                neg_train_genomes.append(idx)
            elif ncbi_genome_category[idx] == 'derived from metagenome':
                neg_test_genomes.append(idx)
    assert len(pos_train_genomes) >= 2 and len(neg_train_genomes) >= 2, f'# genomes for {args.pos_species}: {len(pos_train_genomes)}\t# genomes for {args.neg_species}: {len(neg_train_genomes)}'
    assert len(pos_test_genomes) >= 2 and len(neg_test_genomes) >= 2, f'# genomes for {args.pos_species}: {len(pos_test_genomes)}\t# genomes for {args.neg_species}: {len(neg_test_genomes)}'
    # with open(os.path.join(args.output_dir, 'train_genomes.tsv'), 'w') as f:
    #     random.shuffle(pos_train_genomes)
    #     random.shuffle(neg_train_genomes)
    #     f.write(f'{args.pos_species}\t{genomes[pos_train_genomes[0]][3:]}\t{genomes[pos_train_genomes[0]]}\t{ncbi_genome_category[pos_train_genomes[0]]}\t{ncbi_genome_representation[pos_train_genomes[0]]}\t{gtdb_rep_genome[pos_train_genomes[0]]}\t{gtdb_taxonomy[pos_train_genomes[0]]}\n')
    #     f.write(f'{args.neg_species}\t{genomes[neg_train_genomes[0]][3:]}\t{genomes[neg_train_genomes[0]]}\t{ncbi_genome_category[neg_train_genomes[0]]}\t{ncbi_genome_representation[neg_train_genomes[0]]}\t{gtdb_rep_genome[neg_train_genomes[0]]}\t{gtdb_taxonomy[neg_train_genomes[0]]}\n')
    with open(os.path.join(args.output_dir, 'test_genomes.tsv'), 'w') as f:
        random.shuffle(pos_test_genomes)
        random.shuffle(neg_test_genomes)
        f.write(f'{args.pos_species}\t{genomes[pos_test_genomes[0]][3:]}\t{genomes[pos_test_genomes[0]]}\t{ncbi_genome_category[pos_test_genomes[0]]}\t{ncbi_genome_representation[pos_test_genomes[0]]}\t{gtdb_rep_genome[pos_test_genomes[0]]}\t{gtdb_taxonomy[pos_test_genomes[0]]}\n')
        f.write(f'{args.neg_species}\t{genomes[neg_test_genomes[0]][3:]}\t{genomes[neg_test_genomes[0]]}\t{ncbi_genome_category[neg_test_genomes[0]]}\t{ncbi_genome_representation[neg_test_genomes[0]]}\t{gtdb_rep_genome[neg_test_genomes[0]]}\t{gtdb_taxonomy[neg_test_genomes[0]]}\n')


def AddAlignmentsInfo(args, label_seq, label, genome, genomes_size, out_info):
    blastoutdir = os.path.join(args.output_dir, 'blast', genome)
    # get average percentage identity with train positive genome
    pos_dict_pident = GetMatchRegions(args, os.path.join(blastoutdir, args.pos_label, 'blastn.out'), identity_thr=MIN_IDENTITY, key='subject')
    # get average percentage identity with train negative genome
    neg_dict_pident = GetMatchRegions(args, os.path.join(blastoutdir, args.neg_label, 'blastn.out'), identity_thr=MIN_IDENTITY, key='subject')
    # update sequences with average percentage identity with positive and negative train genomes
    label_seq = AddPctIdentity(pos_dict_pident, label_seq)
    label_seq = AddPctIdentity(neg_dict_pident, label_seq)
    # get coverage of testing genome
    GetGenomeCov(label_seq, genomes_size[genome], label, out_info, 'test')
    return label_seq

def DownloadGenome(args, genome_id):
	if f'{genome_id}' not in os.listdir(args.ncbi_db):
		output_dir = os.path.join(args.ncbi_db, f'{genome_id}')
		os.makedirs(output_dir)
		os.chdir(output_dir)
		# download feature table in gtf if not present
		result = subprocess.run([ncbi_datasets_exec, 'download', 'genome', 'accession', f'{genome_id}', '--include', 'genome,protein'])
		# unzip output folder
		with zipfile.ZipFile('ncbi_dataset.zip', 'r') as zip_ref:
			zip_ref.extractall(os.getcwd())
		os.chdir(args.input_dir)
	else:
		print(f'{genome_id}\tdownload already done')

def PrepareDNASeq(args, genome_id, label, sampling_rate):
    # remove header, plasmids and \n and write sequence to file
    fasta_file = glob.glob(os.path.join(args.ncbi_db, f'{genome_id}/ncbi_dataset/data/{genome_id}/*.fna'))
    assert len(fasta_file) > 0, f'fasta file for {genome_id} not downloaded'
    sequence = ''
    for record in SeqIO.parse(fasta_file[0], "fasta"):
        # remove phages and plasmids
        if 'plasmid' not in record.description and 'Plasmid' not in record.description and 'phage' not in record.description:
            sequence += str(record.seq.rstrip())
    print('genome size', len(sequence))
    # run dnabert prep functions
    if sampling_rate != 1.0:
        new_file_path = os.path.join(args.output_dir, 'dna_sequences', genome_id, f"data_sam_k" + str(args.kmer))
        new_file_path_seq = os.path.join(args.output_dir, 'dna_sequences', genome_id, f"data_sam_seq_k" + str(args.kmer)) 
    else:
        new_file_path = os.path.join(args.output_dir, 'dna_sequences', genome_id, f"data_cut_k" + str(args.kmer))
        new_file_path_seq = os.path.join(args.output_dir, 'dna_sequences', genome_id, f"data_cut_seq_k" + str(args.kmer))
    
    if not os.path.exists(os.path.join(args.output_dir, 'dna_sequences', genome_id)):
        os.makedirs(os.path.join(args.output_dir, 'dna_sequences', genome_id))

    new_file = open(new_file_path, "w")
    new_file_seq = open(new_file_path_seq, "w")

    genome_length = len(sequence)
    vectors = []
    sequences = []
    j = 0
    while j < args.rounds:
        if sampling_rate != 1.0:
            starts, ends = sampling(length=genome_length, kmer=args.kmer, sampling_rate=sampling_rate)
            # sample sequences of same length
            seq_length = []
            vector_length = []
            for i in range(len(starts)):
                new_line = sequence[starts[i]:ends[i]]
                sentence = get_kmer_sentence(new_line, kvalue=args.kmer)
                if ends[i] > genome_length:
                    print('end position above genome size!!!!', len(new_line), starts[i], ends[i], ends[i]-starts[i])
                    print(sentence)
                    sys.exit(1)
                vector_length.append(len(sentence.split(" ")))
                seq_length.append(len(new_line))
                new_file.write(f'{label}\t{sentence}\t{starts[i]}\t{ends[i]}\t{len(sentence.split(" "))}\n')
                new_file_seq.write(f'{label}\t{new_line}\t{starts[i]}\t{ends[i]}\t{len(new_line)}\n')
                vectors.append(f'{label}\t{sentence}\t{starts[i]}\t{ends[i]}\t{len(sentence.split(" "))}\n')
                sequences.append(f'{label}\t{new_line}\t{starts[i]}\t{ends[i]}\t{len(new_line)}\n')
            print(min(seq_length), max(seq_length), statistics.mean(seq_length), statistics.median(seq_length))
            print(min(vector_length), max(vector_length), statistics.mean(vector_length), statistics.median(vector_length))
        else:
            cuts = cut_no_overlap(length=genome_length, kmer=args.kmer)
            start = 0
            seq_length = []
            vector_length = []
            for cut in cuts:
                new_line = sequence[start:start+cut]
                sentence = get_kmer_sentence(new_line, kvalue=args.kmer)
                # sentence = get_kmer_sentence(new_line, kmer=args.kmer)
                vector_length.append(len(sentence.split(" ")))
                seq_length.append(len(new_line))
                end = start + cut
                assert end <= genome_length, f'{end} > {genome_length}'
                # new_file.write(sentence + "\n")
                new_file.write(f'{label}\t{sentence}\t{start}\t{start+cut}\t{len(sentence.split(" "))}\n')
                new_file_seq.write(f'{label}\t{new_line}\t{start}\t{start+cut}\t{len(new_line)}\n')
                vectors.append(f'{label}\t{sentence}\t{start}\t{start+cut}\t{len(sentence.split(" "))}\n')
                sequences.append(f'{label}\t{new_line}\t{start}\t{start+cut}\t{len(new_line)}\n')
                start += cut
            print(min(seq_length), max(seq_length), statistics.mean(seq_length), statistics.median(seq_length))
            print(min(vector_length), max(vector_length), statistics.mean(vector_length), statistics.median(vector_length))
        j += 1
    return vectors, genome_length

def AddPctIdentity(dict_pident, sequences):
    up_sequences = []
    for i in range(len(sequences)):
        start = int(sequences[i].split('\t')[2])
        end = int(sequences[i].split('\t')[3])
        list_pident = []
        for j in range(start, end+1, 1):
            if j in dict_pident:
                list_pident.append(dict_pident[j])
            else:
                list_pident.append(0)
        avg_pident = sum(list_pident)/len(list_pident)
        up_sequences.append(sequences[i].rstrip() + f'\t{avg_pident}\n')
    return up_sequences

def GetMatchRegions(args, input_file, identity_thr=MIN_IDENTITY, key=None):
	# align_coords = []
    dict_pident = {}
    with open(input_file, 'r') as f:
        for count, line in enumerate(f, 1):
            sstart = int(line.rstrip().split(',')[2])
            send = int(line.rstrip().split(',')[3])
            qstart = int(line.rstrip().split(',')[4])
            qend = int(line.rstrip().split(',')[5])
            pident = float(line.rstrip().split(',')[8])
            qseq = line.rstrip().split(',')[9]
            sseq = line.rstrip().split(',')[10]
            if key == 'query':
                for i in range(qstart, qend+1, 1):
                    dict_pident[i] = pident
            elif key == 'subject':
                for i in range(sstart, send+1, 1):
                    dict_pident[i] = pident

			# if pident >= identity_thr:
            # align_coords.append([qstart, qend, pident])

	# return align_coords, query_pident
    return dict_pident

def RunBlast(list_queries, list_labels, output_dir):
    print(list_queries)
    print(list_labels)
    for i in range(len(list_labels)):
        if not os.path.isdir(os.path.join(output_dir, list_labels[i])):
            os.makedirs(os.path.join(output_dir, list_labels[i]))
        # compare target genome with genome of negative label (query)
        result = subprocess.run([blastn_exec, '-query', f'{list_queries[i]}', '-task', 'blastn', '-db', f'{output_dir}/blastdb', '-out', f'{output_dir}/{list_labels[i]}/blastn.out', \
			 '-outfmt', "10 delim=, qseqid sseqid sstart send qstart qend qlen evalue pident qseq sseq sstrand", \
			 '-max_target_seqs', '5', '-num_threads', '1'])

def GetSequences(args, genomes, labels, genomes_size, sequences):
    for i in range(len(genomes)):
        DownloadGenome(args, genomes[i])
        sam_vectors, size = PrepareDNASeq(args, genomes[i], labels[i], 0.5)
        cut_vectors, _ = PrepareDNASeq(args, genomes[i], labels[i], 1.0)
        genomes_size[genomes[i]] = size
        sequences[labels[i]] = sam_vectors + cut_vectors

def GetNumberSequences(sequences, genome_size, min_coverage):
    # get number of sequences to have at least 1x coverage for training 
    sum_bases = 0
    num_seqs = 0
    for s in sequences:
        seq = s.rstrip().split('\t')[1].split(" ")
        original_seq = seq[0]
        for i in range(1, len(seq), 1):
            original_seq += seq[i][-1]
        sum_bases += len(original_seq)
        num_seqs += 1
        if sum_bases / genome_size >= min_coverage:
            break
    
    train_size = round(0.7*num_seqs)
    val_size = num_seqs - train_size

    return train_size, val_size, sum_bases

def GetTrainValData(args, sequences, out_f, label, train_genomes_df=None, label_train_size=None, label_val_size=None):
    random.shuffle(sequences)
    out_f.write(f'{label}\t')
    if args.multiclass:
        if len(sequences) < label_train_size + label_val_size:
            num_extra_seqs = label_train_size + label_val_size - len(sequences)
            add_seqs_indices = [random.randint(0, len(sequences)-1) for i in range(num_extra_seqs)]
            add_seqs = []
            for idx in add_seqs_indices:
                add_seqs.append(sequences[idx])
            sequences += add_seqs
        
        train_size = label_train_size
        val_size = label_val_size
    else:
        if args.bert_step == 'pretraining':
            fasta = train_genomes_df[train_genomes_df[0]==int(label)][2].tolist()[0]
            genome_size = get_genome_size(fasta)
            train_size, val_size, sum_bases = GetNumberSequences(sequences, genome_size, args.min_coverage)
            out_f.write(f'{sum_bases/genome_size}\t')
        else:
            train_size = round(0.7*len(sequences))
            val_size = len(sequences) - train_size

    train_data = sequences[:train_size]
    val_data = sequences[-val_size:]

    out_f.write(f'{train_size}\t{val_size}\n')
    return train_data, val_data

def GetGenomeCov(data, genome_size, label, out_f, datatype): 
    data_cov = {i: 0 for i in range(0, genome_size, 1)}
    for i in range(len(data)):
        start = int(data[i].split('\t')[2])
        end = int(data[i].split('\t')[3])
        for j in range(start, end, 1):
            data_cov[j] += 1
    pct_genome_covered = (sum([1 for v in data_cov.values() if v != 0])/genome_size)*100
    coverage = sum(data_cov.values())/genome_size
    out_f.write(f'{datatype}\t{label}\t{pct_genome_covered}\t{coverage}\n')

def GetGenomeSize(list_fasta, list_labels):
    sizes = {}
    for i in range(len(list_labels)):
        seq = ''
        with open(list_fasta[i], 'r') as f:
            for line in f:
                if line[0] != '>':
                    seq += line.rstrip()
        sizes[list_labels[i]] = len(seq)
    return sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='path to input directory')
    parser.add_argument('--output_dir', type=str, help='path to output file')
    parser.add_argument('--ncbi_db', type=str, help='path to ncbi database built with datasets')
    parser.add_argument('--gtdb_info', type=str, help='path to bac120_metadata_r226.tsv.gz file')
    parser.add_argument('--train_genomes_info', type=str, help='path to train_genomes.tsv file')
    parser.add_argument('--test_genomes_info', type=str, help='path to test_genomes.tsv file')
    parser.add_argument('--dataset', type=str, help='type of dataset to prepare', choices=['train', 'test'])
    parser.add_argument('--bert_step', choices=['pretraining', 'finetuning'])
    parser.add_argument('--kmer', type=int, help='length of kmers')
    parser.add_argument('--rounds', type=int, help='number of times to sample DNA sequences from genomes', default=1)
    parser.add_argument('--min_coverage', type=float, help='minimun coverage of training genome', default=1.5)
    parser.add_argument('--multiclass', action='store_true', default=False)
    parser.add_argument('--num_processes', type=int, help='number of processes to run in parallel')
    parser.add_argument('--pos_species', type=str, help='species acting as the positive class for a binary classifier')
    parser.add_argument('--neg_species', type=str, help='species acting as the negative class for a binary classifier')
    parser.add_argument('--pos_label', type=str, help='labels of species acting as the positive class for a binary classifier')
    parser.add_argument('--neg_label', type=str, help='labels of species acting as the negative class for a binary classifier')
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # train_genomes_df = pd.read_csv(args.train_genomes_info, header=None, sep="\t")
    # train_genomes_df.columns = ['label','genome']
    # pos_train_genome = train_genomes_df[train_genomes_df['label'] == int(args.pos_label)]['genome'].tolist()[0]
    # neg_train_genome = train_genomes_df[train_genomes_df['label'] == int(args.neg_label)]['genome'].tolist()[0]
    labels = [args.neg_label, args.pos_label]
    print('labels', labels)

        # if args.bert_step == 'pretraining' or args.multiclass:
        #     # args.min_coverage == 1.5 for pre-training
        #     if args.multiclass:
        #         # get largest genome in training dataset
        #         genomes_size = {}
        #         for i in range(len(train_genomes_df)):
        #             genomes_size[train_genomes_df[0][i]] = get_genome_size(train_genomes_df[2][i])
                
        #         # get number of sequences of largest genome in the dataset with a coverage of 1x
        #         largest_genome_label = max(genomes_size, key=genomes_size.get)
        #         largest_genome_size = max(genomes_size.values())
        #         print(f'largest genome size: {largest_genome_size}\tlabel: {largest_genome_label}')

        #         label_train_size, label_val_size, _ = GetNumberSequences(sequences[str(largest_genome_label)], largest_genome_size, args.min_coverage)

        #         print(f'largest genome train size: {label_train_size}\tlargest genome val size: {label_val_size}')

        #     all_train_data = []
        #     all_val_data = []
        #     with open(os.path.join(args.output_dir, f'data_info_k{args.kmer}.tsv'), 'w') as out_f:
        #         for l in labels:
        #             GetTrainValData(args, sequences[l], all_train_data, all_val_data, out_f, label=l, train_genomes_df=train_genomes_df, label_train_size=label_train_size, label_val_size=label_val_size)
        #         out_f.write(f'total\t{len(all_train_data)}\t{len(all_val_data)}')

        #     random.shuffle(all_val_data)
        #     random.shuffle(all_train_data)

        #     with open(os.path.join(args.output_dir, f'train_data_k{args.kmer}.tsv'), 'w') as out_f:
        #         out_f.write(''.join(all_train_data))

        #     with open(os.path.join(args.output_dir, f'val_data_k{args.kmer}.tsv'), 'w') as out_f:
        #         out_f.write(''.join(all_val_data))

        # elif args.bert_step == "finetuning":
            # # load gtdb metadata
            # genomes, _, _, _, _, gtdb_taxonomy = get_gtdb_info(args.gtdb_info)
            # genome_to_tax = dict(zip(genomes, gtdb_taxonomy))
    if args.dataset == 'train':
        # SelectGenomes(args)
        train_genomes_df = pd.read_csv(args.train_genomes_info, header=None, sep="\t")
        train_genomes_df.columns = ['label','genome']
        pos_train_genome = train_genomes_df[train_genomes_df['label'] == int(args.pos_label)]['genome'].tolist()[0]
        neg_train_genome = train_genomes_df[train_genomes_df['label'] == int(args.neg_label)]['genome'].tolist()[0]
        genomes = [neg_train_genome, pos_train_genome]
        # Get sequences
        chunk_size = 1
        grouped_labels = [labels[i:i+chunk_size] for i in range(0, len(labels), chunk_size)]
        grouped_genomes = [[neg_train_genome],[pos_train_genome]]
        with mp.Manager() as manager:
            sequences = manager.dict()
            genomes_size = manager.dict()
            processes = [mp.Process(target=GetSequences, args=(args, grouped_genomes[i], grouped_labels[i], genomes_size, sequences)) for i in range(len(grouped_genomes))]
            for p in processes:
                p.start() 
            for p in processes:
                p.join()
            # create BLAST database for training genome of label 1
            pos_fasta = glob.glob(os.path.join(args.ncbi_db, f'{pos_train_genome}/ncbi_dataset/data/{pos_train_genome}/*.fna'))[0]
            blastoutdir = os.path.join(args.output_dir, 'blast', pos_train_genome)
            if not os.path.isdir(blastoutdir):
                os.makedirs(blastoutdir)
            result = subprocess.run([makeblastdb_exec, '-in', f'{pos_fasta}', '-input_type', 'fasta', '-dbtype', 'nucl', '-out', f'{blastoutdir}/blastdb'])
            # BLAST training genomes
            neg_fasta = glob.glob(os.path.join(args.ncbi_db, f'{neg_train_genome}/ncbi_dataset/data/{neg_train_genome}/*.fna'))[0]
            RunBlast([neg_fasta], [args.neg_label], blastoutdir)
            # calculate the number of sequences to sample per label
            num_seq = min([len(sequences[args.pos_label]), len(sequences[args.neg_label])])
            # get sequences
            all_train_data = []
            all_val_data = []
            if not os.path.isdir(os.path.join(args.output_dir, 'train_dataset')):
                os.makedirs(os.path.join(args.output_dir, 'train_dataset'))

            with open(os.path.join(args.output_dir, f'train_dataset/{args.bert_step}_l{args.pos_label}_train_data_info_k{args.kmer}.tsv'), 'w') as out_f:
                for i in range(len(labels)):
                    if labels[i] != args.pos_label:
                        dict_pident = GetMatchRegions(args, os.path.join(blastoutdir, labels[i], 'blastn.out'), identity_thr=MIN_IDENTITY, key='query')
                    else:
                        dict_pident = GetMatchRegions(args, os.path.join(blastoutdir, args.neg_label, 'blastn.out'), identity_thr=MIN_IDENTITY, key='subject')   
                    label_seq = sequences[labels[i]]
                    random.shuffle(label_seq)
                    label_seq = label_seq[:num_seq]
                    label_seq = AddPctIdentity(dict_pident, label_seq)
                    train_data, val_data = GetTrainValData(args, label_seq, out_f, labels[i])
                    GetGenomeCov(train_data, genomes_size[genomes[i]], labels[i], out_f, 'train')
                    GetGenomeCov(val_data, genomes_size[genomes[i]], labels[i], out_f, 'val')
                    all_train_data += train_data
                    all_val_data += val_data
            random.shuffle(all_val_data)
            random.shuffle(all_train_data)
            
            with open(os.path.join(args.output_dir, f'train_dataset/{args.bert_step}_l{args.pos_label}_train_data_k{args.kmer}.tsv'), 'w') as out_f:
                out_f.write(''.join(all_train_data))

            with open(os.path.join(args.output_dir, f'train_dataset/{args.bert_step}_l{args.pos_label}_val_data_k{args.kmer}.tsv'), 'w') as out_f:
                out_f.write(''.join(all_val_data))

    elif args.dataset == 'test':
        SelectGenomes(args)
        sys.exit(1)
        # get testing genomes
        test_genomes_df = pd.read_csv(args.test_genomes_info, header=None, sep="\t")
        test_genomes_df.columns = ['label','genome']
        # get size of testing genomes
        pos_test_genome = test_genomes_df[test_genomes_df['label'] == int(args.pos_label)]['genome'].tolist()[0]
        neg_test_genome = test_genomes_df[test_genomes_df['label'] == int(args.neg_label)]['genome'].tolist()[0]
        genomes = [neg_test_genome, pos_test_genome]
        # Get sequences
        chunk_size = 1
        grouped_labels = [labels[i:i+chunk_size] for i in range(0, len(labels), chunk_size)]
        grouped_genomes = [[neg_test_genome],[pos_test_genome]]
        with mp.Manager() as manager:
            sequences = manager.dict()
            genomes_size = manager.dict()
            processes = [mp.Process(target=GetSequences, args=(args, grouped_genomes[i], grouped_labels[i], genomes_size, sequences)) for i in range(len(grouped_genomes))]
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            print(genomes_size)
            # create BLAST database for testing genomes
            for g in genomes:
                fasta = glob.glob(os.path.join(args.ncbi_db, f'{g}/ncbi_dataset/data/{g}/*.fna'))[0]
                blastoutdir = os.path.join(args.output_dir, 'blast', g)
                if not os.path.isdir(blastoutdir):
                    os.makedirs(blastoutdir)
                result = subprocess.run([makeblastdb_exec, '-in', f'{fasta}', '-input_type', 'fasta', '-dbtype', 'nucl', '-out', f'{blastoutdir}/blastdb'])
                # BLAST genomes
                RunBlast([neg_train_fasta, pos_train_fasta], labels, blastoutdir)

            # only for finetuning
            out_info = open(os.path.join(args.output_dir, f'{args.bert_step}_l{args.pos_label}_test_data_info_k{args.kmer}.tsv'), 'w')
            
            # # get sequences from negative label
            # labels_other = [l for l in labels if l != args.pos_label]
            # print(f'# negative labels: {len(labels_other)}')
            # num_sp_labels = len(labels_other)
            print(len(sequences[args.pos_label]))
            print(len(sequences[args.neg_label]))
            # num_seq = min([len(sequences[args.pos_label]), len(sequences[args.neg_label])])
            # print(f'min num seq: {num_seq}\tpos:{len(sequences[args.pos_label])}\tneg:{len(sequences[args.neg_label])}')
            # num_seq_per_sp = [num // num_sp_labels + (1 if x < num % num_sp_labels else 0) for x in range (num_sp_labels)]
            # print(num, num_sp_labels, len(num_seq_per_sp), num_seq_per_sp[:3])
            if not os.path.isdir(os.path.join(args.output_dir, 'test_dataset')):
                os.makedirs(os.path.join(args.output_dir, 'test_dataset'))
            with open(os.path.join(args.output_dir, f'test_dataset/{args.bert_step}_l{args.pos_label}_test_data_k{args.kmer}.tsv'), 'w') as out_f:
                for i in range(len(labels)):
                    label_seq = sequences[labels[i]]
                    label_seq = AddAlignmentsInfo(args, label_seq, labels[i], genomes[i], genomes_size, out_info)
                    out_f.write(''.join(label_seq))
                    out_info.write(f'{labels[i]}\t{len(label_seq)}\n')


if __name__ == "__main__":
    main()
