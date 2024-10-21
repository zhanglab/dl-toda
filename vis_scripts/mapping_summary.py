import sys
# import numpy as np
import multiprocessing as mp
# import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import math
import statistics
import json


def load_fq_file(fq_file):
    with open(fq_file, 'r') as f:
        content = f.readlines()
    reads_id = [''.join(content[i:i+4]).split('\n')[0][1:] for i in range(0, len(content), 4)]
    return reads_id


def get_dltoda_results(dl_toda_output, reads_id):
    results = defaultdict(list)  # key = read_id, value = prediction score for target label
    with open(dl_toda_output, 'r') as f:
        for count, line in enumerate(f, 0):
            results[reads_id[count]] = [line.rstrip().split('\t')[1], float(line.rstrip().split('\t')[-1])]
    return results


def sum_results(fw_dl_toda_output, fw_fq_file, rv_dl_toda_output, rv_fq_file):
    fw_reads_id = load_fq_file(fw_fq_file)
    rv_reads_id = load_fq_file(rv_fq_file)
    fw_results = get_dltoda_results(fw_dl_toda_output, fw_reads_id)
    rv_results = get_dltoda_results(rv_dl_toda_output, rv_reads_id)

    return {**fw_results, **rv_results}


def extend_cigar(cigar):
    new_cigar = ''
    num = ''
    for i in cigar:
        if i.isnumeric():
            num += i
        else:
            new_cigar += i*int(num)
            num = ''
    return new_cigar


def get_average_pred_score(dict_pred_scores):
    dict_avg_pred_scores = {}
    for k, v in dict_pred_scores.items():
        dict_avg_pred_scores[k] = statistics.mean([x[0] for x in v])
    return list(dict_avg_pred_scores.keys()), list(dict_avg_pred_scores.values())


# def barplot(dict_pred_scores, ref, length_ref, genome_id):
#     # compute average prediction score per position
#     x_values, y_values = get_average_pred_score(dict_pred_scores)
#     print(len(x_values), len(y_values))
#
#     # calculate house coverage
#     # total_sum_bp = sum(y_values)
#     # house_cov = round(float(total_sum_bp) / float(length_ref), 3)
#     # summary = open('/'.join([os.getcwd(), 'coverage_summary.txt']), 'a')
#     # summary.write('\t'.join([genome_id, ref.split(' ')[0], str(house_cov), '\n']))
#     # print('Genome: {0} - Reference: {1} - house coverage: {2}'.format(genome_id, ref.split(' ')[0], house_cov))
#     #
#     plt.clf()
#     plt.bar(x_values, y_values, width=0.2)
#     plt.xticks(np.arange(min(x_values), max(x_values), step=500), rotation=70)
#     # plt.title('{0}'.format(ref))
#     plt.xlabel('Base position')
#     plt.ylabel('Average prediction score')
#     plt.xlim((1, int(length_ref)))
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     # Save plot
#     plt.savefig('Genome-{0}-Sequence-{1}.png'.format(genome_id, ref.split(' ')[0]))


def get_coverage(list_of_reads, dict_cigars, dltoda_results, dict_coverage, dict_pred_scores, dict_pred_labels, process_id):
    # for each reference in the dictionary, create a new dictionary with the
    # number of matches at each position encountered
    # dict_coverage[process_id] = defaultdict(lambda: 0)
    local_dict_coverage = defaultdict(int)
    local_dict_pred_scores = defaultdict(list)
    local_dict_pred_labels = defaultdict(list)

    for j in range(0, len(list_of_reads)):
        read_id = list_of_reads[j]
        read_start = dict_cigars[read_id][0] - 1
        read_cigar = extend_cigar(dict_cigars[read_id][1])
        refmoveset = {'X', 'D', 'N'}
        # refmoveset = {'M', '=', 'X', 'D', 'N'}
        # refnomoveset = {'I', 'S', 'H', 'P'}
        query_pos = 0
        ref_pos = query_pos + read_start

        while query_pos < len(read_cigar):
            if read_cigar[query_pos] in ["=", "M"]:
                local_dict_coverage[ref_pos] += 1
                local_dict_pred_scores[ref_pos].append(dltoda_results[read_id][1])
                local_dict_pred_labels[ref_pos].append(dltoda_results[read_id][0])
            # if read_cigar[query_pos] in refmoveset:
            ref_pos += 1
            query_pos += 1

    dict_coverage[process_id] = local_dict_coverage
    dict_pred_scores[process_id] = local_dict_pred_scores
    dict_pred_labels[process_id] = local_dict_pred_labels

    # compute mean coverage
    # mean_cov = round(sum(dict_coverage.values())/length_ref, 3)

    # return mean_cov


def merge_dict(input_dict):
    output_dict = defaultdict(list)
    for process_id, process_data in input_dict.items():
        for k, v in process_data.items():
            if type(v) == list:
                output_dict[k] += v
            else:
                output_dict[k].append(v)
    return output_dict


def get_summary(dict_pred_scores, dict_coverage, dict_pred_labels, length_ref, genome_id, class_mapping):
    with open(f'{genome_id}-mapping-summary.tsv', 'w') as f:
        for i in range(length_ref):
            if i in dict_pred_scores:
                f.write(f'{i}\t{len(dict_pred_scores[i])}\t{statistics.mean(dict_pred_scores[i])}\t')
                # count the occurrence of predicted labels
                count_labels = Counter(dict_pred_labels[i])
                max_label = [k for k, v in count_labels.items() if v == max(count_labels.values())][0]
                f.write(f'{max_label}\t{class_mapping[str(max_label)]}\n')
            else:
                f.write(f'{i}\t0\tNA\tNA\tNA\n')
            # if i in dict_coverage:
            #     f.write(f'{sum(dict_coverage[i])}\n')
            # else:
            #     f.write('0\n')
            # if i in dict_pred_scores and i not in dict_coverage:
            #     print(f'position: {i}\t{dict_pred_scores[i]}')


def multiprocesses(dict_cigars, dltoda_results, length_ref, ref, num_processes, genome_id, class_mapping):
    reads_id_aligned = list(dict_cigars)
    chunk_size = math.ceil(len(reads_id_aligned) / num_processes)
    reads_id_groups = [reads_id_aligned[i:i + chunk_size] for i in range(0, len(reads_id_aligned), chunk_size)]

    with mp.Manager() as manager:
        dict_coverage = manager.dict()  # key = process id, value = {position: number of reads mapped to that position}
        dict_pred_scores = manager.dict()  # key = process id, value = {position: prediction score of ground truth label}
        dict_pred_labels = manager.dict()  # key = process id, value = {position: predicted label}
        processes = [mp.Process(target=get_coverage, args=(reads_id_groups[x], dict_cigars, dltoda_results, dict_coverage, dict_pred_scores, dict_pred_labels, x)) for x in range(len(reads_id_groups))]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

        # compute mean coverage
        mean_cov = float()
        for coverage_info in dict_coverage.values():
            mean_cov += round(sum(coverage_info.values()) / length_ref, 3)
        print(f'{ref}\tlength: {length_ref}\t# reads: {len(reads_id_aligned)}\tmean coverage: {mean_cov}')

        # update dictionaries
        all_dict_coverage = merge_dict(dict_coverage)
        all_dict_pred_scores = merge_dict(dict_pred_scores)
        all_dict_pred_labels = merge_dict(dict_pred_labels)

        # create summary
        get_summary(all_dict_pred_scores, all_dict_coverage, all_dict_pred_labels, length_ref, genome_id, class_mapping)

        # barplot(all_dict_pred_scores, ref, length_ref, genome_id)


def get_references(content, alignments):
    ref = {}
    for line in content:
        if line.rstrip().split('\t')[0][:3] == '@SQ' and line.rstrip().split('\t')[1].split(':')[1] in alignments:
            ref[line.rstrip().split('\t')[1].split(':')[1]] = int(line.rstrip().split('\t')[2].split(':')[1])
    return ref


def get_alignments(samfile):
    alignments = defaultdict(dict)
    with open(samfile, 'r') as f:
        content = f.readlines()
        for count, i in enumerate(range(len(content)), 1):
            if content[i].rstrip().split('\t')[0][:3] not in ['@PG', '@SQ', '@HD'] and content[i].rstrip().split('\t')[5] != '*':
                if content[i].rstrip().split('\t')[0][-2] != '/':
                    read_id = content[i].rstrip().split('\t')[0] + '/1' if count % 2 == 0 else content[i].rstrip().split('\t')[0] + '/2'
                    alignments[content[i].rstrip().split('\t')[2]][read_id] = [int(content[i].rstrip().split('\t')[3]), content[i].rstrip().split('\t')[5]]
                else:
                    alignments[content[i].rstrip().split('\t')[2]][content[i].rstrip().split('\t')[0]] = [int(content[i].rstrip().split('\t')[3]), content[i].rstrip().split('\t')[5]]

    # get references and their length
    ref = get_references(content[1:], alignments)
    return ref, alignments


if __name__ == "__main__":
    samfile = sys.argv[1]
    fw_dl_toda_output = sys.argv[2]
    rv_dl_toda_output = sys.argv[3]
    fw_fq_file = sys.argv[4]
    rv_fq_file = sys.argv[5]
    num_processes = int(sys.argv[6])
    class_mapping_file = sys.argv[7]

    f = open(class_mapping_file)
    class_mapping = json.load(f)

    dltoda_results = sum_results(fw_dl_toda_output, fw_fq_file, rv_dl_toda_output, rv_fq_file)

    genome_id = samfile.split('/')[-1].split('-')[0]

    ref_info, alignments = get_alignments(samfile)

    # compute mean coverage per reference
    for ref, length_ref in ref_info.items():
        multiprocesses(alignments[ref], dltoda_results, length_ref, ref, num_processes, genome_id, class_mapping)



