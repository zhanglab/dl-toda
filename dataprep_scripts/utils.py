import sys


def load_fq_file(fq_file, num_lines):
    with open(fq_file, 'r') as f:
        content = f.readlines()
        reads = [''.join(content[j:j + num_lines]) for j in range(0, len(content), num_lines)]
        return reads


if __name__ == "__main__":
	input_fq = sys.argv[1]

	reads = load_fq_file(input_fq, 4)
	out_filename = input_fq.split(".")[0] + "-read_count"
	with open(out_filename, "w") as f:
		f.write(f'{len(reads)}\n')

