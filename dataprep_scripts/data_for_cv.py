import os
import argparse
import random

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data', help='data for training and validation', nargs='+', default=[])
	parser.add_argument('--folds', type=int, help='number of folds', default=5)
	parser.add_argument('--type_of_data', type=str, help='type of data', choices=['tsv','fastq'])
	parser.add_argument('--output_dir', type=str, help='path to output directory', default=os.getcwd())
	parser.add_argument('--header', action='store_true', required=('tsv' in sys.argv))
	args = parser.parse_args()

	# load data
	all_data = []
	for datafile in args.data:
		with open(datafile, 'r') as f:
			content = f.readlines()
			if args.header:
				all_data += content[1:]
			else:
				all_data += content

	# shuffle data
	random.shuffle(all_data)
	
	# split data into 5 identical subsets
	examples_per_fold = len(all_data) // args.folds
	print(f'# examples per fold: {examples_per_fold}')
	subsets = [all_data[i:i+examples_per_fold] for i in range(0, len(all_data), examples_per_fold)]

	# create files of data for cross-validation
	for k in range(folds):
		# create output directory for every subset of data
		fold_dir = os.path.join(args.output_dir, f'cv_subset_{k}')
		if not os.path.isdir(fold_dir):
        	os.makedirs(fold_dir)
    	val_data = subsets[k]
    	train_data = [subsets[i] for i in range(k) if i != k]
		with open(os.path.join(fold_dir, 'train.tsv'), 'w') as out_f:
			out_f.write(''.join(train_data))
		with open(os.path.join(fold_dir, 'dev.tsv'), 'w') as out_f:
			out_f.write(''.join(val_data))










if __name__ == "__main__":
	main()