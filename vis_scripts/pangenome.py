import sys
import os
import argparse
from collections import defaultdict
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
from pangenome_utils import *



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--training_fasta', type=str, help='path to training fasta file', required=True)
	parser.add_argument('--testing_fasta', type=str, help='path to testing fasta file', required=True)
	parser.add_argument('--testing_genome', type=str, help='accession id of testing genome', required=True)
	parser.add_argument('--training_genome', type=str, help='accession id of testing genome', required=True)
	parser.add_argument('--output_dir', type=str, help='path to output directory', required=True)
	parser.add_argument('--analysis', help="analyze false positives (FP) or false negatives (FN)", choices=['FP', 'FN'], required=True)
	parser.add_argument('--annotations_dir', type=str, help='path to directory containing gtf annotations files', required=True)
	parser.add_argument('--testing_file', type=str, help='path to fasta/tsv file containing testing reads', required=True)
	parser.add_argument('--prob_threshold', type=float, help='probability score threshold', required=True)
	parser.add_argument('--testing_results', type=str, help='path to file containing testing results', required=True)
	parser.add_argument('--num_processes', type=int, help='number of processes to run in parallel', required=True)
	args = parser.parse_args()

	# define input directory
	input_dir = os.getcwd()

	path_dl_toda = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])

	# get label of training genome
	with open(os.path.join(path_dl_toda, 'data/training_genomes.tsv'), 'r') as f:
		for line in f:
			if line.rstrip().split('\t')[0] == args.training_genome:
				args.train_label = line.rstrip().split('\t')[1]
	with open(os.path.join(path_dl_toda, 'data/testing_genomes.tsv'), 'r') as f:
		for line in f:
			if line.rstrip().split('\t')[0] == args.testing_genome:
				args.test_label = line.rstrip().split('\t')[1]

	assert len(args.train_label) != 0, f'label of training genome {args.training_genome} can not be found'
	assert len(args.test_label) != 0, f'label of testing genome {args.testing_genome} can not be found'

	# create output directories
	args.output_dir = os.path.join(os.getcwd(), args.train_label, args.testing_genome)
	if not os.path.isdir(args.output_dir):
		os.makedirs(args.output_dir)
	if not os.path.isdir(os.path.join(args.output_dir, 'blast')):
		os.makedirs(os.path.join(args.output_dir, 'blast'))
	if not os.path.isdir(os.path.join(args.output_dir, 'Genomes_GTF_missing')):
		os.makedirs(os.path.join(args.output_dir, 'Genomes_GTF_missing'))

	outfile_sum = open(os.path.join(args.output_dir, f'{args.testing_genome}_summary.tsv'), 'w')
	
	# get dltoda taxonomy
	with open(os.path.join(path_dl_toda, 'data/dl_toda_taxonomy.tsv'), 'r') as f:
		args.dl_toda_tax = {line.rstrip().split('\t')[0]: line.rstrip().split('\t')[1] for line in f.readlines()}

	# verify that the genomes investigated only have one chromosome
	testing_records, training_records = CheckGenomes(args)

	# get reads in testing set fasta file
	test_readid_to_read, test_sequence_length, test_ordered_reads_id = LoadFnaFile(args.testing_file)

	# get reads in testing set
	if args.testing_file[-3:] == 'fna':
		test_readid_to_read, test_sequence_length, test_ordered_reads_id = LoadFnaFile(args.testing_file)
	elif args.testing_file[-3:] == 'tsv':
		test_readid_to_read, test_sequence_length, test_ordered_reads_id = LoadTsvFile(args.testing_file)
	print(args.analysis)
	if args.analysis == 'FP':
		incorrect_seq, correct_seq, incorrect_cs, correct_cs = GetFPTNReads(args, test_ordered_reads_id, test_sequence_length, outfile_sum)
	elif args.analysis == 'FN':
		incorrect_seq, correct_seq, incorrect_cs, correct_cs = GetFNTPReads(args, test_ordered_reads_id, test_sequence_length, outfile_sum)
	print('train label',args.train_label)
	# create fasta file with incorrect and correct testing reads
	with open(os.path.join(args.output_dir, f'{args.testing_genome}_test_reads.fna'), 'w') as outf:
		for k, v in test_readid_to_read.items():
			if k in incorrect_seq or k in correct_seq:
				outf.write(f'>{k}\n{v}\n')
	print(f'incorrect_seq: {len(incorrect_seq)}')
	print(f'correct_seq: {len(correct_seq)}')
	# blast testing reads to testing genome
	RunBlast(args, os.path.join(args.output_dir, 'blast', 'test_reads_test_genome'), os.path.join(args.output_dir, f'{args.testing_genome}_test_reads.fna'), subject=[args.testing_fasta], outfilename=f'{args.output_dir}/blast/test_reads_test_genome/all_test_pos_test_blastn.out')
	# get mapping of false and true positives to testing genome
	incorrect_alignments = GetReadsAlignments(incorrect_seq, f'{args.output_dir}/blast/test_reads_test_genome/all_test_pos_test_blastn.out', test_sequence_length, os.path.join(args.output_dir, f'blast/test_reads_test_genome/incorrect_{args.prob_threshold}_mapping_info.tsv'))
	correct_alignments = GetReadsAlignments(correct_seq, f'{args.output_dir}/blast/test_reads_test_genome/all_test_pos_test_blastn.out', test_sequence_length, os.path.join(args.output_dir, f'blast/test_reads_test_genome/correct_{args.prob_threshold}_mapping_info.tsv'))
	
	# get annotations info
	test_annot_info, _ = GetAnnotInfo(args, args.testing_genome, input_dir)
	# get genes and functions associated with incorrect and correct predictions
	scores, incorrect_genes, correct_genes = GetGenes(args, test_annot_info, incorrect_alignments, correct_alignments, test_sequence_length, test_readid_to_read, len(testing_records[0].seq), incorrect_cs, correct_cs)

	GetReadsForAttentions(args, correct_alignments, incorrect_alignments, test_readid_to_read)

	avg_pct_identity, ani, test_strain, train_strain = CircosPlot(args, scores, testing_records[0].seq, training_records[0].seq, args.testing_fasta, args.training_fasta, incorrect_alignments, correct_alignments, \
			 os.path.join(args.output_dir, f'{args.testing_genome}_{args.prob_threshold}_circos.png'))

	train_species = args.dl_toda_tax[args.train_label].split(';')[0]
	train_genus = args.dl_toda_tax[args.train_label].split(';')[1]
	test_species = args.dl_toda_tax[args.test_label].split(';')[0]
	test_genus = args.dl_toda_tax[args.test_label].split(';')[1]

	with open(os.path.join(args.output_dir, f'{args.testing_genome}_incorrect_shared_genes_{args.prob_threshold}.tsv'), 'w') as f:
		for k, v in incorrect_genes.items():
			f.write(f'{args.test_label}\t0\t{args.testing_genome}\t{test_strain}\t{" ".join(test_species)}\t{" ".join(test_genus)}\t{args.train_label}\t{args.training_genome}\t')
			f.write(f'{train_strain}\t{train_species}\t{train_genus}\t{avg_pct_identity}\t{ani}\t{k}\t{v[0]}\t{v[1]}\t{v[2]}\t{v[3]}\t{v[4]}\t')
			if test_annot_info[k][0] == 'protein_coding':
				f.write(f'{test_annot_info[k][0]}\t{test_annot_info[k][4]}\t{test_annot_info[k][5]}\t{test_annot_info[k][6]}\n')
			else:
				f.write(f'{test_annot_info[k][0]}\t{test_annot_info[k][4]}\tNA\n')

	with open(os.path.join(args.output_dir, f'{args.testing_genome}_correct_unique_genes_{args.prob_threshold}.tsv'), 'w') as f:
		for k, v in correct_genes.items():
			f.write(f'{args.test_label}\t0\t{args.testing_genome}\t{test_strain}\t{" ".join(test_species)}\t{" ".join(test_genus)}\t{args.train_label}\t{args.training_genome}\t')
			f.write(f'{train_strain}\t{train_species}\t{train_genus}\t{avg_pct_identity}\t{ani}\t{k}\t{v[0]}\t{v[1]}\t{v[2]}\t{v[3]}\t{v[4]}\t')
			if test_annot_info[k][0] == 'protein_coding':
				f.write(f'{test_annot_info[k][0]}\t{test_annot_info[k][4]}\t{test_annot_info[k][5]}\t{test_annot_info[k][6]}\n')
			else:
				f.write(f'{test_annot_info[k][0]}\t{test_annot_info[k][4]}\tNA\n')

	if args.analysis == 'FP':
		# get taxa of FP reads
		if len(incorrect_sequences) > 0:
			incorrect_labels = set([s.split('|')[1] for s in list(incorrect_sequences)])
			incorrect_taxa = defaultdict(int)
			for label in incorrect_labels:
				# get fp sequences of label
				label_sequences = set([seq_id for seq_id in incorrect_sequences if seq_id.split('|')[1] == label])
				# monitor number of sequences per label
				incorrect_taxa[label] = len(label_sequences)
			incorrect_taxa_sorted = dict(sorted(incorrect_taxa.items(), key=lambda item: item[1], reverse=True))
			with open(os.path.join(args.output_dir, f'{args.testing_genome}_{args.prob_threshold}_fp_taxa.tsv'), 'w') as f:
				for k, v in incorrect_taxa_sorted.items():
					f.write(f'{k}\t{args.dl_toda_tax[k]}\t{v}\n')




	

