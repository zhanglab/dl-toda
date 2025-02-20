import sys
import os
import glob
import argparse
import math
import zipfile
import subprocess
import random
import statistics
from collections import defaultdict
from pycirclize import Circos, config
from Bio.SeqFeature import SeqFeature, FeatureLocation
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
from dataprep_scripts.utils import load_fq_file
from vis_scripts.parse_samfile import LoadData, GetCoverageOfSample
from pygenomeviz.parser import Fasta
from pygenomeviz.utils import load_example_fasta_dataset, ColorCycler, interpolate_color
from pygenomeviz.align import AlignCoord, Blast
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
ColorCycler.set_cmap("Set1")

QUERY_TRACK_SIZE = 5
MIN_IDENTITY = 70
TICKS_INTERVAL = 100000
bowtie2_build_exec = "/modules/uri_apps/software/Bowtie2/2.4.5-GCC-11.3.0/bin/bowtie2-build"
bowtie2_exec = "/modules/uri_apps/software/Bowtie2/2.4.5-GCC-11.3.0/bin/bowtie2"
ncbi_datasets_exec = "/work/pi_yingzhang_uri_edu/ccres/tools/datasets"

# set seed
seed = 42
# set the global python random seed
random.seed(seed)


def GetFNOtherInfo(args, pos_test_alignments, neg_train_alignments, annot_info):
	taxa = defaultdict(int)
	list_reads_id = []
	with open(os.path.join(args.output_dir, f'fn_pos_test_neg_train_{args.prob_threshold}_summary.tsv'), 'w') as f:
		for readid, data in neg_train_alignments.items():
			taxa[data[0]] += 1
			list_reads_id.append(readid)
			if readid in pos_test_alignments:
				genes = set()
				for pos in range(pos_test_alignments[readid][1], pos_test_alignments[readid][2]+1, 1):
					for gene_id, annot in annot_info.items():
						if pos >= annot[0] and pos <= annot[1]:
							genes.add(annot[3])
				f.write(f"{readid}\t{data[0]}")
				for g in list(genes):
					f.write(f'\t{g}')
				f.write('\n')
				
	taxa_sorted = dict(sorted(taxa.items(), key=lambda item: item[1], reverse=True))
	most_mapped_taxon = ''
	with open(os.path.join(args.output_dir, f'fn_pos_test_neg_train_{args.prob_threshold}_mapped_taxa.tsv'), 'w') as f:
		for count, (k, v) in enumerate(taxa_sorted.items()):
			if count == 0:
				most_mapped_taxon = k
			f.write(f'{k}\t{args.dl_toda_tax[k]}\t{v}\n')

	return most_mapped_taxon, list_reads_id


def CreateFqFile(genes_of_interest, alignments, readid_to_read, filename):
	reads_of_interest = set()
	for readid, data in alignments.items():
		for pos in range(data[1], data[2]+1, 1):
			for gene_id, annot in genes_of_interest.items():
				if pos >= annot[0] and pos <= annot[1]:
					reads_of_interest.add(readid_to_read[readid])
	with open(filename, 'w') as f:
		f.write(''.join(list(reads_of_interest)))


def GetAnnotInfo(args, genome_id, input_dir):
	annot_info = defaultdict(list)
	if f'{genome_id}_gtf' not in os.listdir(args.annotations_dir):
		annot_output_dir = os.path.join(args.annotations_dir, f'{genome_id}_gtf')
		os.makedirs(annot_output_dir)
		os.chdir(annot_output_dir)
		# download feature table in gtf if not present
		result = subprocess.run([ncbi_datasets_exec, 'download', 'genome', 'accession', f'{genome_id}', '--include', 'gtf'])
		# unzip output folder
		with zipfile.ZipFile('ncbi_dataset.zip', 'r') as zip_ref:
			zip_ref.extractall(os.getcwd())
		os.chdir(input_dir)
	else:
		print(f'{genome_id}\tdownload already done')

	
	annot_file = glob.glob(os.path.join(args.annotations_dir, f'{genome_id}_gtf/ncbi_dataset/data/{genome_id}/genomic.gtf'))
	if len(annot_file) != 0:
		print(annot_file)
		with open(annot_file[0], 'r') as f:
			content = f.readlines()
			for i in range(5,len(content)-1,1):
				# if content[i].rstrip().split('\t')[2] == 'gene':
				begin = int(content[i].rstrip().split('\t')[3])
				end = int(content[i].rstrip().split('\t')[4])
				strand = content[i].rstrip().split('\t')[6]
				gene_id = ''
				gene = ''
				for e in content[i].rstrip().split('\t')[8].split(';'):
					e = e.replace('"', '')
					if 'product' in e:
						gene = ' '.join(e.split(' ')[2:])
					if 'gene_id' in e:
						gene_id = e.split(' ')[1]
				if gene_id == '':
					assert gene_id != None, 'gene id should not be unknown'
				if gene != '':
					annot_info[gene_id] = [begin, end, strand, gene]
	print('ANNOT INFO')
	print(annot_info)
	return annot_info

def GetPosOfInterest(args, annot_info, alignments, sequence_length):
	# get count and length of fn sequences per mapped position on the genome investigated
	# positions_count = defaultdict(int)
	positions_seq_length = defaultdict(list)
	genes_of_interest = defaultdict(list)
	readid_w_gene = defaultdict(list)
	for readid, data in alignments.items():
		for pos in range(data[1], data[2]+1, 1):
			for gene_id, annot in annot_info.items():
				if pos >= annot[0] and pos <= annot[1]:
					# positions_count[begin] += 1
					genes_of_interest[gene_id] = annot_info[gene_id]
					readid_w_gene[readid] = [data[1], data[2]]				
			positions_seq_length[pos].append(sequence_length[readid])

	print(f'# genes of interest: {len(genes_of_interest)}')
	if len(readid_w_gene) != len(alignments):
		reads_not_associated_w_genes = []
		with open(os.path.join(args.output_dir, 'test_reads_wo_gene.tsv'), 'w') as f:
			for readid, data in alignments.items():
				if readid not in readid_w_gene:
					print(readid, data)
					f.write(f'{readid}\t{data[0]}\t{data[1]}\t{data[2]}\t{data[3]}\t{data[4]}\n')
					reads_not_associated_w_genes.append(readid)
	else:
		print('all reads were found a gene')

	# sort positions_count by values
	# genes_of_interest = defaultdict(list)
	# positions_count_sorted = dict(sorted(positions_count.items(), key=lambda item: item[1], reverse=True))
	# for count, (k, v) in enumerate(positions_count_sorted.items(), 1):
	# 	print(k, v, annot_info[k])
	# 	genes_of_interest[k] = annot_info[k]
		# if count == 20:
		# 	break
	
	return positions_seq_length, genes_of_interest, reads_not_associated_w_genes


def GetAlignmentsInfo(sequences, samfile, sequence_length, seq_to_labels, outfilename=None):
	alignments = defaultdict(list)
	mapped_reads_id = []
	unmapped_reads_id = []
	with open(samfile, 'r') as f:
		for line in f:
			if line.rstrip().split('\t')[0][:3] not in ['@PG', '@SQ', '@HD']:
				read_id = line.rstrip().split('\t')[0]
				if read_id in sequences:
					if line.rstrip().split('\t')[5] != '*':
						seq_id = line.rstrip().split('\t')[2]
						seq_label = seq_to_labels[seq_id]
						start_pos = int(line.rstrip().split('\t')[3])
						mapping_score = int(line.rstrip().split('\t')[4])
						alignments[read_id] = [seq_label, start_pos, start_pos+sequence_length[read_id], mapping_score, seq_id]
						mapped_reads_id.append(read_id)
					else:
						unmapped_reads_id.append(read_id)
	if outfilename:
		with open(outfilename, 'w') as f:
			f.write(f'# mapped reads:\t{len(mapped_reads_id)}\n# unmapped reads:\t{len(unmapped_reads_id)}')

	return alignments


def GetSeqLength(sequences_id, sequence_length, type):
	if len(sequences_id) != 0:
		seq_length_info = [sequence_length[s] for s in sequences_id]
		print(f'{type}\tmean: {statistics.mean(seq_length_info)}\tmedian: {statistics.median(seq_length_info)}\tmax: {max(seq_length_info)}\tmin: {min(seq_length_info)}')

def FNCircosPlot(args, most_mapped_taxon, most_mapped_reads_id, fn_alignments_pos_test, tp_alignments_pos_test, cds_to_show, fn_positions_seq_length, fn_not_associated_w_genes, outfilepath, outfigpath):

	# load data from training and testing genomes of label 1
	target_fasta = Fasta(args.test_genomes_info[args.label][1]) # ref/subject --> target --> testing genome
	# comp_fasta_list = list(map(Fasta, [args.train_genomes_info[args.label][1], args.train_genomes_info[most_mapped_taxon][1]])) # query --> training genome
	comp_fasta_list = list(map(Fasta, [args.train_genomes_info[args.label][1]])) # query --> training genome
	# print(target_fasta.__dict__)

	# Initialize circos instance
	circos = Circos(
	    sectors=target_fasta.get_seqid2size(),
	    # space=0 if len(target_fasta.get_seqid2size()) == 1 else 2,
		space=8,
	)
	print('define space', len(target_fasta.get_seqid2size()))
	# circos.text(f"{target_fasta.name}\n({target_fasta.full_genome_length:,} bp)", size=13)
	print(f"{target_fasta.name}\n({target_fasta.full_genome_length:,} bp)\n{target_fasta.full_genome_length}")

	outf = open(outfilepath, 'w')

	min_r_pos = 100
	for sector in circos.sectors:
		# Setup outer track
		outer_track = sector.add_track((min_r_pos-0.3, min_r_pos))
		outer_track.axis(fc="black")
		outer_track.xticks_by_interval(TICKS_INTERVAL, label_formatter=lambda v: f"{v/1000000:.1f} Mb")
		outer_track.xticks_by_interval(50000, tick_length=1, show_label=False)
		# create tracks for genomics features
		# f_cds_track = sector.add_track((min_r_pos-5, min_r_pos))
		# f_cds_track.axis(fc="lightgrey", ec="none", alpha=0.5)
		# min_r_pos -= 5
		# r_cds_track = sector.add_track((min_r_pos-5, min_r_pos))
		# r_cds_track.axis(fc="lightgrey", ec="none", alpha=0.5)
		# min_r_pos -= 5
		# Plot forward/reverse strand CDS
		min_r_pos -= 1
		cds_track = sector.add_track((min_r_pos-5, min_r_pos))
		features = []
		for gene_id in cds_to_show.keys():
			if cds_to_show[gene_id][2] == 'plus':
				location = FeatureLocation(start=cds_to_show[gene_id][0], end=cds_to_show[gene_id][1], strand=+1)
				feature = SeqFeature(location=location, qualifiers={"gene_id": [gene_id], "gene_name": [cds_to_show[gene_id][3]], "strand": ["plus"]})
		# 		f_cds_track.genomic_features(feature, plotstyle="arrow", fc="salmon", lw=0.5)
				cds_track.genomic_features(feature, plotstyle="arrow", fc="salmon")
			else:
				location = FeatureLocation(start=cds_to_show[gene_id][0], end=cds_to_show[gene_id][1], strand=-1)
				feature = SeqFeature(location=location, qualifiers={"gene_id": [gene_id], "gene_name": [cds_to_show[gene_id][3]], "strand": ["minus"]})
		# 		r_cds_track.genomic_features(feature, plotstyle="arrow", fc="skyblue", lw=0.5)
				cds_track.genomic_features(feature, plotstyle="arrow", fc="skyblue")
			features.append(feature)

		# Add regions not associated with genes and mapped by FN reads
		min_r_pos -= 5
		ukn_track = sector.add_track((min_r_pos-5, min_r_pos))
		for readid, data in fn_alignments_pos_test.items():
			if readid in fn_not_associated_w_genes:
				ukn_track.rect(data[1], data[2], color="red")
				if readid in most_mapped_reads_id:
					print(f'read id mapped to {most_mapped_taxon}: {readid}\t{data}')
		

		# Plot gene label if it exists
		# labels, label_pos_list = [], []
		for feature in features:
			start = int(feature.location.start)
			end = int(feature.location.end)
			label_pos = (start + end) / 2
			gene_id = feature.qualifiers.get("gene_id", [None])[0]
			label = feature.qualifiers.get("gene_name", [None])[0]
			strand = feature.qualifiers.get("strand", [None])[0]
			outf.write(f'{gene_id}\t{strand}\t{feature.qualifiers.get("gene_name", [None])[0]}\t{start}\t{end}\n')
			print(f'{gene_id}\t{strand}\t{feature.qualifiers.get("gene_name", [None])[0]}\t{start}\t{end}\n')
			if label == None:
				continue
			# if gene_id is not None:
			# 	labels.append(gene_id)
			# 	label_pos_list.append(label_pos)
			# f_cds_track.annotate(label_pos, label, label_size=7)

		# f_cds_track.xticks(label_pos_list, labels, label_size=8, label_orientation="vertical")

		# if sector.size >= TICKS_INTERVAL:
		# 	r_cds_track.xticks_by_interval(
		# 		TICKS_INTERVAL,
		# 		outer=False,
		# 		label_formatter=lambda v: f"{v/1000000:.1f} Mb"
		# 	)

	# Blast genome comparison & plot match blocks
	min_r_pos -= 5
	comp_name2color = {}
	colors = ["black", "gray"]
	for idx, comp_fasta in enumerate(comp_fasta_list):
		align_coords = Blast([target_fasta, comp_fasta]).run()
		align_coords = AlignCoord.filter(align_coords, identity_thr=MIN_IDENTITY)
		# color = ColorCycler()
		comp_name2color[comp_fasta.name] = colors[idx]
		min_r_pos -= QUERY_TRACK_SIZE
		print(min_r_pos, min_r_pos + QUERY_TRACK_SIZE)
		for sector in circos.sectors:
			sector.add_track((min_r_pos, min_r_pos + QUERY_TRACK_SIZE), r_pad_ratio=0.1)	
		for ac in align_coords:
			print(ac.query_start, ac.query_end)
			track = circos.get_sector(ac.query_name).tracks[-1] # Last added track in sector
			rect_color = interpolate_color(colors[idx], v=ac.identity, vmin=MIN_IDENTITY) # type: ignore
			track.rect(ac.query_start, ac.query_end, color=rect_color)

	for sector in circos.sectors:
		# define x-axis vector
		genome_pos = list(range(target_fasta.full_genome_length))

		# add track for TP reads
		min_r_pos -= 10
		tp_track = sector.add_track((min_r_pos, min_r_pos + 8), r_pad_ratio=0.1)
		tp_track.axis()
		pos_tp_count = [0]*target_fasta.full_genome_length
		for data in tp_alignments_pos_test.values():
			for pos in range(data[1], data[2], 1):
				pos_tp_count[pos-1] +=1
		y_values = list(range(min(pos_tp_count), max(pos_tp_count), 3))
		y_labels = list(map(str, y_values))
		tp_track.yticks(y_values, y_labels)
		tp_track.line(genome_pos, pos_tp_count, color="blue")
			# tp_track.rect(data[1], data[2], color="orange", lw=0.1)
		print(min_r_pos, min_r_pos + 10)
		print(f'added TP track')

		# add tracks for FN reads that didn't map to any training genomes 
		min_r_pos -= 10
		fn_track_1 = sector.add_track((min_r_pos, min_r_pos + 8), r_pad_ratio=0.1)
		fn_track_1.axis()
		pos_fn_1_count = [0]*target_fasta.full_genome_length
		for readid, data in fn_alignments_pos_test.items():
			if readid not in most_mapped_reads_id:
				for pos in range(data[1], data[2], 1):
					pos_fn_1_count[pos-1] +=1
		y_values = list(range(min(pos_fn_1_count), max(pos_fn_1_count), 2))
		y_labels = list(map(str, y_values))
		fn_track_1.yticks(y_values, y_labels)
		fn_track_1.line(genome_pos, pos_fn_1_count, color="red")
			# fn_track.rect(data[1], data[2], color="red", lw=0.1)
		print(f'added FN track')

		# add tracks for FN reads that were mapped to a taxon from label 0
		min_r_pos -= 10
		fn_track_2 = sector.add_track((min_r_pos, min_r_pos + 8), r_pad_ratio=0.1)
		fn_track_2.axis()
		pos_fn_2_count = [0]*target_fasta.full_genome_length
		for readid, data in fn_alignments_pos_test.items():
			if readid in most_mapped_reads_id:
				for pos in range(data[1], data[2], 1):
					pos_fn_2_count[pos-1] +=1
		y_values = list(range(min(pos_fn_2_count), max(pos_fn_2_count), 2))
		y_labels = list(map(str, y_values))
		fn_track_2.yticks(y_values, y_labels)
		fn_track_2.line(genome_pos, pos_fn_2_count, color="orange")
			# fn_track.rect(data[1], data[2], color="red", lw=0.1)
		print(f'added FN track')

		# add tracks for average sequence length of FN reads
		min_r_pos -= 10
		seq_track = sector.add_track((min_r_pos, min_r_pos + 8), r_pad_ratio=0.1)
		seq_track.axis()
		avg_seq_length = []
		for i in range(1, target_fasta.full_genome_length+1, 1):
			if i in fn_positions_seq_length:
				avg_seq_length.append(sum(fn_positions_seq_length[i])/len(fn_positions_seq_length[i]))
			else:
				avg_seq_length.append(0)
		print(f'{len(avg_seq_length)}\n{statistics.mean(avg_seq_length)}\n{statistics.median(avg_seq_length)}\n{max(avg_seq_length)}\n{min(avg_seq_length)}')
		y_values = list(range(min([math.ceil(x) for x in avg_seq_length]), max(math.ceil(x) for x in avg_seq_length), 200))
		y_labels = list(map(str, y_values))
		seq_track.yticks(y_values, y_labels)
		# seq_track.bar(avg_seq_length, pos_seq_length, color="green", lw=0.5)
		seq_track.line(genome_pos, avg_seq_length, color="green")
		print(f'added sequence length track')

	# save figure
	# Enable annotation text adjustment (Default)
	# config.ann_adjust.enable = True
	fig = circos.plotfig()
	fig.savefig(outfigpath, dpi=300)



if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--pos_test_neg_train', type=str, help='path to sam file with mapping of testing sequences from label 1 to training genomes from label 0 - get info about FN sequences')
	parser.add_argument('--pos_test_pos_test', type=str, help='path to sam file with mapping of testing sequences from label 1 to testing genome from label 1')
	parser.add_argument('--pos_train_pos_train', type=str, help='path to sam file with mapping of training sequences from 1 to training genome from label 1 - get info about coverage')
	# parser.add_argument('--pos_train_fasta', type=str, help='path to training genome fasta file from label 1')
	# parser.add_argument('--pos_test_fasta', type=str, help='path to testing genome fasta file from label 1')
	parser.add_argument('--training_fasta', type=str, help='path to file containing list of fasta files')
	parser.add_argument('--annotations_dir', type=str, help='path to directory containing gtf annotations files')
	parser.add_argument('--testing_fq_file', type=str, help='path to fastq file containing all testing reads (label 1 and 0)')
	parser.add_argument('--testing_fasta', type=str, help='path to file containing path to fasta files of training genomes')
	parser.add_argument('--label', type=str, help='label of species investigated', required=True)
	parser.add_argument('--sequences_info', type=str, help='path to file mapping labels of species in model to sequences id of all sequences in training set')
	parser.add_argument('--prob_threshold', type=float, help='probability score threshold', required=True)
	parser.add_argument('--rank', type=str, help='taxonomic rank investigated', choices=['species','genus','family','order','class', 'phylum'])
	parser.add_argument('--testing_results', type=str, help='path to file containing testing results')
	args = parser.parse_args()

	input_dir = os.getcwd()

	# create output directory
	args.output_dir = os.path.join(os.getcwd(), args.label)
	if not os.path.isdir(args.output_dir):
		os.makedirs(args.output_dir)
	
	# get dltoda taxonomy
	path_dl_toda_tax = '/'.join(
                os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]) + '/data/dl_toda_taxonomy.tsv'
	with open(path_dl_toda_tax, 'r') as in_f:
		content = in_f.readlines()
		args.dl_toda_tax = {line.rstrip().split('\t')[0]: line.rstrip().split('\t')[1] for line in content}

	# retrieve accession and fasta files of testing and tranining genomes associated with each label
	with open(args.testing_fasta, 'r') as f:
		content = f.readlines()
		args.test_genomes_info = {line.rstrip().split('\t')[0]: [line.rstrip().split('\t')[1], line.rstrip().split('\t')[2]] for line in content}

	with open(args.training_fasta, 'r') as f:
		content = f.readlines()
		args.train_genomes_info = {line.rstrip().split('\t')[0]: [line.rstrip().split('\t')[1], line.rstrip().split('\t')[2]] for line in content}

	# get reads in testing set fastq file
	sequences = load_fq_file(args.testing_fq_file, 4)
	readid_to_read = {line.split('\n')[0][1:]: line for line in sequences}
	sequence_length = {line.split('\n')[0][1:]: len(line.split('\n')[1]) for line in sequences}

	# get FN, FP and TP sequences
	fn_sequences = set()
	fp_sequences = set()
	tp_sequences = set()

	with open(args.testing_results, 'r') as f:
		for count, line in enumerate(f):
			prob = float(line.rstrip().split('\t')[2])
			if prob >= args.prob_threshold:
				if line.rstrip().split('\t')[0] == '1' and line.rstrip().split('\t')[1] == '0':
					fn_sequences.add(sequences[count].split('\n')[0][1:])
				if line.rstrip().split('\t')[0] == '0' and line.rstrip().split('\t')[1] == '1':
					fp_sequences.add(sequences[count].split('\n')[0][1:])
				if line.rstrip().split('\t')[0] == '1' and line.rstrip().split('\t')[1] == '1':
					tp_sequences.add(sequences[count].split('\n')[0][1:])

	print(f'#FN for label {args.label}: {len(fn_sequences)}')
	print(f'#FP for label {args.label}: {len(fp_sequences)}')
	print(f'#TP for label {args.label}: {len(tp_sequences)}')
	GetSeqLength(list(fn_sequences), sequence_length, f'label {args.label} testing FN sequences')
	GetSeqLength(list(fp_sequences), sequence_length, f'label {args.label} testing FP sequences')
	GetSeqLength(list(tp_sequences), sequence_length, f'label {args.label} testing TP sequences')

	# get association between sequences in training set and labels
	with open(args.sequences_info, 'r') as f:
		content = f.readlines()
		seq_to_labels = {line.rstrip().split('\t')[0]: line.rstrip().split('\t')[1] for line in content}

	# get coverage of training genome with training sequences for label 1
	train_ref_info, train_train_alignments = LoadData(args.pos_train_pos_train)
	training_genome_size = train_ref_info[0][1]
	train_train_dict_coverage, train_train_reads_info = GetCoverageOfSample(train_train_alignments[train_ref_info[0][0]], training_genome_size, label=args.label)
	train_train_coverage = [train_train_dict_coverage[i] for i in range(training_genome_size)]
	positions_w_zero = 0
	for i in range(len(train_train_coverage)):
		if train_train_coverage[i] == 0:
			positions_w_zero += 1
	print(f'train-train coverage: {positions_w_zero}\ntraining_genome_size: {training_genome_size}')
	print(f'mean: {statistics.mean(train_train_coverage)}\tmedian: {statistics.median(train_train_coverage)}\tmin: {min(train_train_coverage)}\tmax: {max(train_train_coverage)}')
	
	# do FN analysis
	# get mapping of false negatives to testing genome from label 1
	fn_alignments_pos_test = GetAlignmentsInfo(fn_sequences, args.pos_test_pos_test, sequence_length, seq_to_labels, os.path.join(args.output_dir, f'fn_pos_test_pos_test_{args.prob_threshold}_mapping_info.tsv'))
	
	# get annotations info
	pos_test_annot_info = GetAnnotInfo(args, args.test_genomes_info[args.label][0], input_dir)
	fn_positions_seq_length, fn_genes_of_interest, fn_not_associated_w_genes = GetPosOfInterest(args, pos_test_annot_info, fn_alignments_pos_test, sequence_length)
	print(len(fn_genes_of_interest))

	# get mapping of false negatives to training genomes from other species
	fn_alignments_pos_neg_train = GetAlignmentsInfo(fn_sequences, args.pos_test_neg_train, sequence_length, seq_to_labels, os.path.join(args.output_dir, f'fn_pos_test_neg_train_{args.prob_threshold}_mapping_info.tsv'))
	# get taxonomy of mapped training genomes and taxon with most reads mapped
	most_mapped_taxon, most_mapped_reads_id = GetFNOtherInfo(args, fn_alignments_pos_test, fn_alignments_pos_neg_train, pos_test_annot_info)
	# get mapping of true positives to testing genome from label 1
	tp_alignments_pos_test = GetAlignmentsInfo(tp_sequences, args.pos_test_pos_test, sequence_length, seq_to_labels, os.path.join(args.output_dir, f'tp_pos_test_pos_test_{args.prob_threshold}_mapping_info.tsv'))

	# create fastq files with FN and TP reads mapping positions of interest on the testing genome
	CreateFqFile(fn_genes_of_interest, fn_alignments_pos_test, readid_to_read, os.path.join(args.output_dir, f'{args.label}_{args.prob_threshold}_fn_reads.fq'))
	CreateFqFile(fn_genes_of_interest, tp_alignments_pos_test, readid_to_read, os.path.join(args.output_dir, f'{args.label}_{args.prob_threshold}_tp_reads.fq'))

	# create circos plot with FN reads info
	FNCircosPlot(args, most_mapped_taxon, most_mapped_reads_id, fn_alignments_pos_test, tp_alignments_pos_test, fn_genes_of_interest, fn_positions_seq_length, fn_not_associated_w_genes, os.path.join(args.output_dir, f'{args.label}_{args.prob_threshold}_fn_genes.tsv'), os.path.join(args.output_dir, f'{args.label}_{args.prob_threshold}_fn_circos.png'))

	# # do FP analysis
	# # map reads in testing dataset to their corresponding genome
	# fp_labels = set([s.split('|')[1] for s in list(fp_sequences)])
	# fp_taxa = defaultdict(int)
	# print(f'# labels: {len(fp_labels)}')
	# outf = open(os.path.join(args.output_dir, f'{args.label}_{args.prob_threshold}_fp_neg_genes.tsv'), 'w')
	# outf_problem = open(os.path.join(args.output_dir, f'{args.label}_{args.prob_threshold}_fp_neg_genomes_missing.tsv'), 'w')
	# for label in fp_labels:
	# 	label_testing_fasta = args.test_genomes_info[label][1]
	# 	label_testing_genome = args.test_genomes_info[label][0]
		
	# 	mapping_output_dir = f'{args.output_dir}/mapping/label0/testing-genome/{label}'
	# 	if not os.path.isdir(mapping_output_dir):
	# 		os.makedirs(mapping_output_dir)
		
	# 	if 'results.sam' not in os.listdir(mapping_output_dir):
	# 		# build bowtie2 index 
	# 		result = subprocess.run([bowtie2_build_exec, '--threads', '1', f'{label_testing_fasta}', f'{mapping_output_dir}/ref'])
	# 		# map reads
	# 		result = subprocess.run([bowtie2_exec, '-x', f'{mapping_output_dir}/ref', '-U', f'{args.testing_fq_file}', '-S', f'{mapping_output_dir}/results.sam' ])
	# 	else:
	# 		print(f'{label}\talignment already done')
		
	# 	# get fp sequences of label
	# 	label_sequences = [seq_id for seq_id in fp_sequences if seq_id.split('|')[1] == label]
	# 	print(label, len(label_sequences))

	# 	# get alignments info
	# 	samfile = os.path.join(mapping_output_dir, 'results.sam')
	# 	fp_alignments = GetAlignmentsInfo(label_sequences, samfile, sequence_length, seq_to_labels)
		
	# 	print(label_testing_genome)
	# 	# get gene associated with sequences
	# 	neg_test_annot_info = GetAnnotInfo(args, label_testing_genome, input_dir)
	# 	if len(neg_test_annot_info) > 0:
	# 		neg_genes_mapped = defaultdict(int)
	# 		neg_genes_strand = defaultdict(str)
	# 		for seq_id in label_sequences:
	# 			start_mapping = fp_alignments[seq_id][1]
	# 			end_mapping = fp_alignments[seq_id][2]
	# 			for pos in range(start_mapping, end_mapping+1, 1):
	# 				for begin in neg_test_annot_info.keys():
	# 					if pos >= begin and pos <= neg_test_annot_info[begin][0]:
	# 						neg_genes_mapped[neg_test_annot_info[begin][2]] += 1
	# 						neg_genes_strand[neg_test_annot_info[begin][2]] = neg_test_annot_info[begin][1]

	# 		neg_genes_mapped_sorted = dict(sorted(neg_genes_mapped.items(), key=lambda item: item[1], reverse=True))
	# 		for k, v in neg_genes_mapped_sorted.items():
	# 			outf.write(f'{label}\t{k}\t{v}\t{neg_genes_strand[k]}\n')
	# 	else:
	# 		outf_problem.write(f'{label}\t{label_testing_genome}\n')
			
	# 	# monitor number of sequences per label
	# 	fp_taxa[label] = len(label_sequences)
	
	# fp_taxa_sorted = dict(sorted(fp_taxa.items(), key=lambda item: item[1], reverse=True))
	# with open(os.path.join(args.output_dir, f'{args.label}_{args.prob_threshold}_fp_neg_taxa.tsv'), 'w') as f:
	# 	for k, v in fp_taxa_sorted.items():
	# 		f.write(f'{k}\t{args.dl_toda_tax[k]}\t{v}\n')
	
	# fp_reads = [readid_to_read[r] for r in fp_sequences]
	# with open(os.path.join(args.output_dir, f'{args.label}_{args.prob_threshold}_fp_reads.fq'), 'w') as f:
	# 	f.write(''.join(fp_reads))
	

	







