import sys
import numpy as np


def main():
	val_data = sys.argv[1]

	with open(val_data, 'r') as f:
		content = f.readlines()
		loss = [i.rstrip().split('\t')[2] for i in content]

	min_value = np.inf
	epoch_min_value = 1
	for i, v in enumerate(loss, 1):
		if v < min_value:
			min_value = v
			epoch_min_value = i
	
	print(min_value, epoch_min_value)


if __name__ == "__main__":
	main()