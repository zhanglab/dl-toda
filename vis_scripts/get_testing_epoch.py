import sys
import os
import numpy as np


def main():
	val_data = sys.argv[1]
	output_dir = sys.argv[2]

	with open(os.path.join(val_data, 'validation_data_rnd_1.tsv'), 'r') as f:
		content = f.readlines()
		loss = [float(i.rstrip().split('\t')[2]) for i in content]

	min_value = np.inf
	epoch_min_value = 1
	for i, v in enumerate(loss, 1):
		if v < min_value:
			min_value = v
			epoch_min_value = i
	
	print(min_value, epoch_min_value)
	with open(os.path.join(output_dir, 'testing_epoch.tsv'), 'a') as f:
		f.write(f'{epoch_min_value}\t{min_value}\n')


if __name__ == "__main__":
	main()