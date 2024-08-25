import sys
import os
from pycirclize import Circos
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
print(sys.path)
from dataprep_scripts.utils import load_fq_file
from collections import defaultdict


def prep_test_results(testing_output, alignment_sum, reads_id, label, ref_length):
	# get testing results
	with open(testing_output, 'r') as f:
		content = f.readlines()
		test_results = dict(zip(reads_id,[i.rstrip() for i in content]))

	# get alignment info
	with open(alignment_sum, 'r') as f:
		content = f.readlines()
		map_info = {i.rstrip().split('\t')[0]: int(i.rstrip().split('\t')[1]) for i in content}

	pos_label = [0]*ref_length
	neg_label = [0]*ref_length
	pos_conf_scores = [0.0]*ref_length
	neg_conf_scores = [0.0]*ref_length
	for r in reads_id:
		if r.split('|')[1] == label:
			# check if read mapped to the genome
			if r in map_info:
				# get start position where the read maps to the target genome
				start_pos = map_info[r] - 1
				# get predicted label
				pred_label = int(test_results[r].rstrip().split('\t')[1])
				cs = float(test_results[r].rstrip().split('\t')[2])
				# add info
				for i in range(start_pos, start_pos + 250, 1):
					if pred_label == 1:
						pos_label[i] += 1
						pos_conf_scores[i] += cs
					elif pred_label == 0:
						neg_label[i] += 1
						neg_conf_scores[i] += cs

	# get average of cs
	for i in range(ref_length):
		if pos_label[i] != 0:
			pos_conf_scores[i] = round(pos_conf_scores[i]/pos_label[i], 2)
		if neg_label[i] != 0:
			neg_conf_scores[i] = round(neg_conf_scores[i]/neg_label[i], 2)

	return pos_label, neg_label, pos_conf_scores, neg_conf_scores


def main():
	fq_file = sys.argv[1]
	testing_output = sys.argv[2]
	genome_cov = sys.argv[3]
	alignment_sum = sys.argv[4]
	label = sys.argv[5]
	output_dir = sys.argv[6]

	# get reads id
	num_lines = 4
	reads = load_fq_file(fq_file, num_lines)
	reads_id = [r.split("\n")[0][1:] for r in reads]
	print(f'# reads: {len(reads_id)}\t{len(reads)}\t{reads_id[0]}')

	# get coverage of target genome per position
	with open(genome_cov, 'r') as f:
		content = f.readlines()
		pos_coverage = [int(i.rstrip().split('\t')[1]) for i in content]


	pos_label, neg_label, pos_conf_scores, neg_conf_scores = prep_test_results(testing_output, alignment_sum, reads_id, label, len(pos_coverage))

	print(f'{len(pos_coverage)}\t{len(pos_label)}\t{len(neg_label)}\t{len(pos_conf_scores)}\t{len(neg_conf_scores)}')

	# initialize a single circos sector
	sectors = {'genome': len(pos_coverage)}
	circos = Circos(sectors=sectors)


	# Plot bar
	base_positions = range(0,len(pos_coverage),1)
	print(len(pos_coverage.values()))
	for sector in circos.sectors:
	    bar_track = sector.add_track((15, 40), r_pad_ratio=0.1)
	    bar_track.axis()
	    bar_track.bar(base_positions, pos_coverage)
	circos.savefig(os.path.join(output_dir, f'sum_circos.png'))




if __name__ == "__main__":
	main()