import sys
import os
import glob
import math
import zipfile
import subprocess
import random
import statistics
import numpy as np
from Bio import SeqIO, SeqUtils
from Bio.SeqFeature import SeqFeature, FeatureLocation
from collections import defaultdict
from pycirclize import Circos, config
from pygenomeviz.parser import Fasta
from pygenomeviz.utils import load_example_fasta_dataset, ColorCycler, interpolate_color
from pygenomeviz.align import AlignCoord, Blast
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

ColorCycler.set_cmap("Set1")

# QUERY_TRACK_SIZE = 5
MIN_IDENTITY = 70
TICKS_INTERVAL = 500000
blastn_exec = "/modules/uri_apps/software/BLAST+/2.15.0-gompi-2023a/bin/blastn"
makeblastdb_exec = "/modules/uri_apps/software/BLAST+/2.15.0-gompi-2023a/bin/makeblastdb"
ncbi_datasets_exec = "/work/pi_yingzhang_uri_edu/ccres/tools/datasets"

# set seed
seed = 42

# set the global python random seed
random.seed(seed)


def GetMatchRegions(input_file, identity_thr=MIN_IDENTITY):
	align_coords = []
	query_pident = {}
	with open(input_file, 'r') as f:
		for count, line in enumerate(f, 1):
			sstart = int(line.rstrip().split(',')[2])
			send = int(line.rstrip().split(',')[3])
			qstart = int(line.rstrip().split(',')[4])
			qend = int(line.rstrip().split(',')[5])
			pident = float(line.rstrip().split(',')[8])
			qseq = line.rstrip().split(',')[9]
			sseq = line.rstrip().split(',')[10]
			for i in range(qstart, qend+1, 1):
				query_pident[i] = pident

			if pident >= identity_thr:
				align_coords.append([qstart, qend, pident])

	return align_coords, query_pident


def RunBlast(output_dir, query, num_processes, subject=None, db=False, outfilename=None, sam=False):
	if not os.path.isdir(output_dir):
		os.makedirs(output_dir)
	if db:
		sys.executable = blastn_exec
		process = subprocess.run([sys.executable, '-query', f'{query}', '-db', '/datasets/bio/ncbi-db/2025-01-26/nt', '-out', \
			f'{output_dir}/blast/test_fp_blastn.out', '-outfmt', "10 delim=, qseqid sseqid evalue pident sstart send qstart qend length ssciname stitle", \
			'-max_target_seqs', '1', '-num_threads', f'{num_processes}'])
	else:
		if len(subject) > 1:
			# put all training genomes into one fasta file
			if not os.path.exists(os.path.join(output_dir, 'all_training_genomes.fna')):
				with open(os.path.join(output_dir, 'all_training_genomes.fna'), 'w') as outf:
					for count, fasta in enumerate(subject, 1):
						print(f'{count}\t{fasta}')
						with open(fasta, 'r') as inf:
							outf.write(inf.read())
				input_fasta = os.path.join(output_dir, 'all_training_genomes.fna')
		else:
			input_fasta = subject[0]

		print(input_fasta)
		# create database
		result = subprocess.run([makeblastdb_exec, '-in', f'{input_fasta}', '-input_type', 'fasta', '-dbtype', 'nucl', '-out', f'{output_dir}/blastdb'])
		
		# align reads to database or fasta file
		if sam:
			result = subprocess.run([blastn_exec, '-query', f'{query}', '-db', f'{output_dir}/blastdb', '-out', f'{outfilename}', \
			 	'-outfmt', "17", '-max_target_seqs', '1', '-num_threads', f'{num_processes}'])
		else:
			# task = 'megablast'
			task = 'blastn'
			result = subprocess.run([blastn_exec, '-query', f'{query}', '-task', f'{task}', '-db', f'{output_dir}/blastdb', '-out', f'{outfilename}', \
			 '-outfmt', "10 delim=, qseqid sseqid sstart send qstart qend qlen evalue pident qseq sseq sstrand", \
			 '-max_target_seqs', '5', '-num_threads', f'{num_processes}'])

def GetGenomesInfo(fasta):
	with open(fasta, 'r') as f:
		content = f.readline()
	strain = ' '.join([e for e in content[0].rstrip().split(',')[0].split(' ')[1:] if e not in ['chromosome', 'strain']])
	
	return strain

def CircosPlot(correct_seq, incorrect_seq, correct_genes, incorrect_genes, training_fasta, testing_fasta, testing_genome, output_dir, num_processes):

    # load data from training and testing genomes of label 1
    query_fasta = Fasta(testing_fasta) # query --> testing genome
    ref_fasta = Fasta(training_fasta) # ref/subject --> training genome

    # Initialize circos instance
    circos = Circos(
        sectors=query_fasta.get_seqid2size(),
        # space=0 if len(ref_fasta.get_seqid2size()) == 1 else 2,
        space=10,
    )

    train_strain = GetGenomesInfo(training_fasta)
    test_strain = GetGenomesInfo(testing_fasta)
    circos.text(f'{test_strain}\n{query_fasta.full_genome_length:,} bp\n(testing genome)', size=9, r=22)

    with open(os.path.join(output_dir, f'{testing_genome}_genomes_length.tsv'), 'w') as f:
        f.write(f'Testing genome:\t{query_fasta.name}\t{query_fasta.full_genome_length}\n')
        f.write(f'Training genome:\t{ref_fasta.name}\t{ref_fasta.full_genome_length}\n')

    min_r_pos = 100
    for sector in circos.sectors:
        # Setup outer track
        outer_track = sector.add_track((min_r_pos-0.3, min_r_pos))
        outer_track.axis(fc="black")
        outer_track.xticks_by_interval(TICKS_INTERVAL, label_formatter=lambda v: f"{v/1000000:.1f} Mb",)
        min_r_pos -= 1
        outer_track.xticks_by_interval(100000, tick_length=1, show_label=False)

    # Blast genome comparison & plot match blocks
    # store percentage identity between matching regions
    percent_identity = []
    # run blast
    RunBlast(os.path.join(output_dir, 'blast', testing_genome, 'test_train_genomes'), testing_fasta, num_processes, subject=[training_fasta], outfilename=f'{output_dir}/blast/{testing_genome}/test_train_genomes/test_train_genomes_blastn.out')
    align_coords, query_pident = GetMatchRegions(f'{output_dir}/blast/{testing_genome}/test_train_genomes/test_train_genomes_blastn.out', identity_thr=MIN_IDENTITY)

    # count the number of identical positions across the aligned regions
    identical_positions = 0
    for sector in circos.sectors:
        blast_track = sector.add_track((min_r_pos-5, min_r_pos), r_pad_ratio=0.1)
        min_r_pos -= 5	
        for ac in align_coords:
            percent_identity.append(ac[2])
            identical_positions += (ac[2]/100*(ac[1]-ac[0]))
            rect_color = interpolate_color("black", v=ac[2], vmin=MIN_IDENTITY)
            blast_track.rect(ac[0], ac[1], color=rect_color)

	# get stats on percentage identity
    avg_pct_identity = round(identical_positions/query_fasta.full_genome_length*100,2)
    ani = round(statistics.mean(percent_identity), 2)
    with open(os.path.join(output_dir, f'{testing_genome}_pct_identity_matching_regions.tsv'), 'w') as f:
        f.write(f'# identical positions\t{identical_positions}\npercentage identity\t{avg_pct_identity}%\n')
        f.write(f'Stats on aligned regions\nmean\t{statistics.mean(percent_identity)}\nmedian\t{statistics.median(percent_identity)}\nmin\t{min(percent_identity)}\nmax\t{max(percent_identity)}')
    
    for sector in circos.sectors:
        # define x-axis vector for the next tracks
        genome_pos = list(range(query_fasta.full_genome_length))

        # # add track for scores
        min_r_pos -= 5
        # scores_track = sector.add_track((min_r_pos-10, min_r_pos), r_pad_ratio=0.1)
        # scores_track.axis(ec="deeppink")
        # y_values = list(range(math.floor(min(scores)), math.ceil(max(scores))+1, 1))
        # y_labels = list(map(str, y_values))
        # scores_track.yticks(y_values, y_labels)
        # scores_track.line(genome_pos, scores, color="deeppink")
        # print(f'added score track')
        # add track for correct and incorrect classification
        # if len(correct_genes) > 0:
        # min_r_pos -= 13
        # Setup track for forward and reverse strand CDS
        f_cds_track = sector.add_track((min_r_pos-10, min_r_pos), r_pad_ratio=0.1)
        f_cds_track.axis(fc="lightgrey", ec="none", alpha=0.5)
        r_cds_track = sector.add_track((min_r_pos-15, min_r_pos-5), r_pad_ratio=0.1)
        r_cds_track.axis(fc="lightgrey", ec="none", alpha=0.5)
        
        # for each egne define a score: 
        # Plot fw and rev strand CDS
        # for gene_id, gene_info in correct_genes.items():
        outfile = open(os.path.join(output_dir, 'testing_genes_pident_training_genome.tsv'), 'w')
        # get all the genes
        scores = {}
        list_genes = list(set(list(correct_genes.keys()) + list(incorrect_genes.keys())))
        for gene_id in list_genes:
            if gene_id != 'NA':
                if gene_id not in correct_genes:
                    c_gene_num = 0
                    c_gene_length = 0
                else:
                    gene_start = correct_genes[gene_id][0][3]
                    gene_end = correct_genes[gene_id][0][4]
                    c_gene_num = len(correct_genes[gene_id])
                    c_gene_length = sum([seq[1] - seq[0] for seq in correct_genes[gene_id]])
                if gene_id not in incorrect_genes:
                    i_gene_num = 0
                    i_gene_length = 0
                else:
                    gene_start = incorrect_genes[gene_id][0][3]
                    gene_end = incorrect_genes[gene_id][0][4]
                    i_gene_num = len(incorrect_genes[gene_id])
                    i_gene_length = sum([seq[1] - seq[0] for seq in incorrect_genes[gene_id]])

                gene_score = ((c_gene_length - i_gene_length)/(c_gene_length + i_gene_length))
                
                pident_pos = []
                for i in range(gene_start, gene_end+1, 1):
                    if i in query_pident:
                        pident_pos.append(query_pident[i])
                    else:
                        pident_pos.append(0)
                avg_pident = round(sum(pident_pos)/len(pident_pos),3)
                scores[gene_id] = gene_score
                outfile.write(f'{gene_id}\t{avg_pident}\t{c_gene_num}\t{i_gene_num}\t{c_gene_length}\t{i_gene_length}\t{gene_score}\n')
            
        # create a color palette
        list_genes = list(scores.keys())
        print(f'# genes: {len(list_genes)}')
        score_values = [scores[k] for k in list_genes]
        v_min = min(score_values)
        v_max = max(score_values)
        # get a color palette from matplotlib
        cmap = plt.get_cmap('viridis')
        # normalize colors based on our values
        norm = mcolors.Normalize(vmin=v_min, vmax=v_max)
        normalized_score_values = norm(score_values)
        rgba_colors = cmap(normalized_score_values)
        value_to_color = dict(zip(normalized_score_values, rgba_colors))

        # plot genes
        for idx, gene_id in enumerate(list_genes):
            if gene_id != 'NA':
                if gene_id in correct_genes:
                    gene_start = correct_genes[gene_id][0][3]
                    gene_end = correct_genes[gene_id][0][4]
                    gene_strand = correct_genes[gene_id][0][5]
                elif gene_id in incorrect_genes:
                    gene_start = incorrect_genes[gene_id][0][3]
                    gene_end = incorrect_genes[gene_id][0][4]
                    gene_strand = incorrect_genes[gene_id][0][5]
                gene_strand = 1 if gene_strand == '+' else -1
                feature = SeqFeature(FeatureLocation(gene_start, gene_end, strand=gene_strand),type="CDS")
                if gene_strand == 1:
                    f_cds_track.genomic_features(feature, plotstyle="arrow", fc=value_to_color[normalized_score_values[idx]], lw=0.5)
                elif gene_strand == -1:
                    r_cds_track.genomic_features(feature, plotstyle="arrow", fc=value_to_color[normalized_score_values[idx]], lw=0.5)

		# # Plot GC skew
		# min_r_pos -= 11
		# gcskew_track = sector.add_track((min_r_pos-5, min_r_pos))
		# pos_list, gcskews = GetGCSkew(test_record_seq)
		# positive_gcskews = np.where(gcskews > 0, gcskews, 0)
		# negative_gcskews = np.where(gcskews < 0, gcskews, 0)
		# abs_max_gcskew = np.max(np.abs(gcskews))
		# vmin, vmax = -abs_max_gcskew, abs_max_gcskew
		# gcskew_track.fill_between(
		# 	pos_list, positive_gcskews, 0, vmin=vmin, vmax=vmax, color="grey"
		# )
		# gcskew_track.fill_between(
		# 	pos_list, negative_gcskews, 0, vmin=vmin, vmax=vmax, color="limegreen"
		# )

		# # Plot GC content
		# min_r_pos -= 5
		# gc_content_track = sector.add_track((min_r_pos-5, min_r_pos))
		# pos_list, gc_content, test_genome_gc_content = GetGCContent(test_record_seq)
		# gc_content_updated = gc_content - test_genome_gc_content
		# positive_gc_content = np.where(gc_content_updated > 0, gc_content_updated, 0)
		# negative_gc_content = np.where(gc_content_updated < 0, gc_content_updated, 0)
		# abs_max_gc_content = np.max(np.abs(gc_content_updated))
		# vmin, vmax = -abs_max_gc_content, abs_max_gc_content
		# gc_content_track.fill_between(
		# 	pos_list, positive_gc_content, 0, vmin=vmin, vmax=vmax, color="black"
		# )
		# gc_content_track.fill_between(
		# 	pos_list, negative_gc_content, 0, vmin=vmin, vmax=vmax, color="deeppink"
		# )
		
		# # report GC content of train and test genomes
		# _, _, train_genome_gc_content = GetGCContent(train_record_seq)
		# with open(os.path.join(output_dir, f'{testing_genome}_GC_content.tsv'), 'w') as f:
		# 	f.write(f'Testing genome:\t{test_genome_gc_content}\n')
		# 	f.write(f'Training genome:\t{train_genome_gc_content}')

	# Save figure
	# Enable annotation text adjustment (Default)
	# config.ann_adjust.enable = True
    fig = circos.plotfig()

    # Create a ScalarMappable only for the colorbar
    sm = ScalarMappable(cmap=cmap)
    sm.set_clim(vmin=vmin, vmax=vmax)

    # Add colorbar (legend for heatmap)
    cbar = fig.colorbar(sm, ax=circos.ax, fraction=0.046, pad=0.04)
    cbar.set_label("Score")
	# Add legend
    handles = []
    handles += [
        Patch(color='black', label=f'{train_strain}\n{ref_fasta.full_genome_length:,} bp (training genome) - {avg_pct_identity}% - {ani}%'),
    ]

	# handles += [
	# 	Line2D([], [], color='blue', label='Positive GC Skew', marker="^", ms=6, ls="None"),
	# 	Line2D([], [], color='gold', label='Negative GC Skew', marker="v", ms=6, ls="None"),
	# 	Line2D([], [], color='darkviolet', label='Positive GC Content', marker="^", ms=6, ls="None"),
	# 	Line2D([], [], color='orangered', label='Negative GC Content', marker="v", ms=6, ls="None")
	# 	]
    _ = circos.ax.legend(handles=handles, bbox_to_anchor=(0.5, 0.475), loc="center", fontsize=8)
    fig.savefig(os.path.join(output_dir, 'circos.png'), dpi=300)

def GetGenes(annot_info, seq_start, seq_end):
    target_gene_id = 'NA'
    target_data = []
    for gene_id, data in annot_info.items():
        gene_start_pos = data[1]
        gene_end_pos = data[2]
        if (seq_start <= gene_start_pos and seq_end >= gene_start_pos) \
            or (seq_start <= gene_end_pos and seq_end >= gene_end_pos) \
            or (seq_start >= gene_start_pos and seq_end <= gene_end_pos) \
            or (seq_start <= gene_start_pos and seq_end >= gene_end_pos):
            target_gene_id = gene_id
            target_data = data
            break
    return target_gene_id, target_data

def GetAnnotInfo(genome_id, input_dir, annotations_dir, output_dir):
	if f'{genome_id}_gtf' not in os.listdir(annotations_dir):
		annot_output_dir = os.path.join(annotations_dir, f'{genome_id}_gtf')
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

	annot_file = glob.glob(os.path.join(annotations_dir, f'{genome_id}_gtf/ncbi_dataset/data/{genome_id}/genomic.gtf'))
	if len(annot_file) == 0:
		f = open(os.path.join(output_dir, 'Genomes_GTF_missing', f'{genome_id}.txt'), 'w')
		f.close()
		return {}
	else:
		genes_type = defaultdict(str)
		annot_info = defaultdict(list)
		locus_tags_info = defaultdict(list)
		with open(annot_file[0], 'r') as f:
			content = f.readlines()
			for i in range(5,len(content)-1,1):
				begin = int(content[i].rstrip().split('\t')[3])
				end = int(content[i].rstrip().split('\t')[4])
				strand = content[i].rstrip().split('\t')[6]
				gene_id = ''
				gene = ''
				biotype = ''
				function = ''
				old_locus_tag = ''
				protein_id = ''
				for e in content[i].rstrip().split('\t')[8].split(';'):
					e = e.replace('"', '')
					# get all go_function entries and choose go_function with the most details
					if 'go_function' in e:
						fn = e.split('|')[0].split(' ')[2:]
						if len(fn) > len(function):
							function = ' '.join(fn)
					if 'product' in e:
						gene = ' '.join(e.split(' ')[2:])
					if 'gene_id' in e:
						gene_id = e.split(' ')[1]
					if 'gene_biotype' in e:
						biotype = e.split(' ')[2]
					if 'old_locus_tag' in e:
						old_locus_tag = e.split(' ')[2]
					if 'protein_id' in e:
						protein_id = e.split(' ')[2]

				if content[i].rstrip().split('\t')[2] == 'gene':
					genes_type[gene_id] = biotype
					locus_tags_info[gene_id] = [begin, end, old_locus_tag, strand]
				elif content[i].rstrip().split('\t')[2] == 'CDS' and genes_type[gene_id] == 'protein_coding':
					if function == '':
						function = gene
					annot_info[gene_id] = ['protein_coding', begin, end, strand, gene, function, protein_id]
				elif content[i].rstrip().split('\t')[2] == 'transcript' and genes_type[gene_id] == 'tRNA':
					annot_info[gene_id] = ['tRNA', begin, end, strand, gene]
				elif content[i].rstrip().split('\t')[2] == 'transcript' and genes_type[gene_id] == 'rRNA':
					annot_info[gene_id] = ['rRNA', begin, end, strand, gene]
				
				assert gene_id != '', 'gene id should not be unknown'
		
		with open(os.path.join(output_dir, f'{genome_id}_genes.tsv'), 'w') as f:
			num_proteins = len([k for k, v in annot_info.items() if v[0] == 'protein_coding'])
			num_rrna = len([k for k, v in annot_info.items() if v[0] == 'rRNA'])
			num_trna = len([k for k, v in annot_info.items() if v[0] == 'tRNA'])
			f.write(f'# genes\t{len(annot_info)}\n# protein coding genes\t{num_proteins}\n# rRNA coding genes\t{num_rrna}\n# tRNA coding genes\t{num_trna}\n')


		return annot_info, locus_tags_info