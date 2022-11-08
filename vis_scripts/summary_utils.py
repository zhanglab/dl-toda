import os
import glob
import pandas as pd

def fill_out_cm(args, predictions, ground_truth, confidence_scores, r_name, r_index):
    ground_truth_taxa = set([ground_truth[i].split(';')[r_index] for i in range(len(ground_truth))])
    predictions_taxa = set([predictions[i].split(';')[r_index] for i in range(len(predictions))])
    predictions_taxa.add('unclassified')
    # create empty confusion matrix with ground truth as columns and predicted taxa as rows
    cm = pd.DataFrame(columns = ground_truth_taxa, index = predictions_taxa)
    # fill out table with zeros
    for c in ground_truth_taxa:
        cm[c] = 0
    # fill out confusion matrix with number of reads
    for i in range(len(predictions)):
        true_taxon = ground_truth[i].split(';')[r_index]
        if confidence_scores[i] > args.cutoff:
            tool_taxon = predictions[i].split(';')[r_index]
        else:
            print(confidence_scores[i], predictions[i], ground_truth[i])
            tool_taxon = 'unclassified'
        cm.loc[tool_taxon, true_taxon] += 1

    return cm

def get_metrics(args, cm, r_name, r_index):
    taxa_in_dl_toda = set([v.split(';')[r_index] for v in args.dl_toda_tax.values()])
    with open(os.path.join(args.output_dir, f'{r_name}-metrics.tsv'), 'w') as out_f:
        out_f.write('true taxon\tpredicted taxon\t#reads\tprecision\trecall\tF1\tTP\tFP\tFN\n')
        ground_truth = list(cm.columns)
        predicted_taxa = list(cm.index)
        correct_predictions = 0
        classified_reads = 0
        unclassified_reads = 0
        problematic_reads = 0
        total_num_reads = 0
        for true_taxon in ground_truth:
            # get number of reads in testing dataset for given taxon
            num_reads = sum([cm.loc[i, true_taxon] for i in predicted_taxa])
            if true_taxon != 'na':
                if true_taxon in taxa_in_dl_toda:
                    if true_taxon in predicted_taxa:
                        predicted_taxon = true_taxon
                        classified_reads += sum([cm.loc[i, true_taxon] for i in predicted_taxa if i != 'unclassified'])
                        other_true_taxa = [i for i in ground_truth if i != true_taxon]
                        true_positives = cm.loc[predicted_taxon, true_taxon]
                        correct_predictions += true_positives
                        false_positives = sum([cm.loc[predicted_taxon, i] for i in other_true_taxa])
                        # add unclassified to include in the count of FN
                        false_negatives = sum([cm.loc[i, true_taxon] for i in predicted_taxa if i not in (predicted_taxon, 'unclassified')])
                        precision = float(true_positives)/(true_positives+false_positives) if true_positives+false_positives > 0 else 0
                        recall = float(true_positives)/(true_positives+false_negatives) if true_positives+false_negatives > 0 else 0
                        f1_score = float(2 * (precision * recall)) / (precision + recall) if precision + recall > 0 else 0
                        out_f.write(f'{true_taxon}\t{predicted_taxon}\t{num_reads}\t{precision}\t{recall}\t{f1_score}\t{true_positives}\t{false_positives}\t{false_negatives}\n')
                    else:
                        if args.zeros:
                            print(f'{true_taxon} has a precision/recall/F1 scores equal to 0')
                            out_f.write(f'{true_taxon}\tna\t{num_reads}\t0\t0\t0\t0\t0\t{num_reads}\n')
                        unclassified_reads += sum([cm.loc[i, true_taxon] for i in predicted_taxa])
                        # unclassified_reads += sum([cm.loc[i, true_taxon] for i in predicted_taxa if i not in ('unclassified', 'na')])
                else:
                    print(f'{true_taxon} with {num_reads} reads is not in {args.tool} model')
                    problematic_reads += sum([cm.loc[i, true_taxon] for i in predicted_taxa if i not in ('unclassified', 'na')])
            else:
                print(f'ground truth unknown: {true_taxon}\t{num_reads}')
                problematic_reads += sum([cm.loc[i, true_taxon] for i in predicted_taxa if i not in ('unclassified', 'na')])

            total_num_reads += num_reads

        if 'unclassified' in predicted_taxa:
            unclassified_reads += sum([cm.loc['unclassified', i] for i in ground_truth])
        if 'na' in predicted_taxa:
            unclassified_reads += sum([cm.loc['na', i] for i in ground_truth])

        out_f.write(f'{correct_predictions}\t{cm.to_numpy().sum()}\t{classified_reads}\t{problematic_reads}\t{unclassified_reads}\t {problematic_reads+unclassified_reads+classified_reads}\t{total_num_reads}\n')

        accuracy_whole = round(correct_predictions/cm.to_numpy().sum(), 5) if cm.to_numpy().sum() > 0 else 0
        accuracy_classified = round(correct_predictions/classified_reads, 5) if classified_reads > 0 else 0
        accuracy_w_misclassified = round(correct_predictions/(classified_reads+unclassified_reads), 5) if (classified_reads+unclassified_reads) > 0 else 0
        out_f.write(f'Accuracy - whole dataset: {accuracy_whole}\n')
        out_f.write(f'Accuracy - classified reads only: {accuracy_classified}\n')
        out_f.write(f'Accuracy - classified and unclassified reads: {accuracy_w_misclassified}')


def combine_cm(args, all_cm, rank):
    excel_files = glob.glob(os.path.join(args.input_dir, f'*-{rank}-confusion-matrix.xlsx'))
    print(rank, len(excel_files))
    df_list = []
    columns = []
    rows = []
    files = []
    for x in excel_files:
        df = pd.read_excel(x, index_col=0, sheet_name=None)
        if rank in df.keys():
            df_list.append(df[rank])
            columns += df[rank].columns.tolist()
            rows += df[rank].index.tolist()
            files.append(x)

    if len(df_list) != 0:
        predicted_taxa = set(rows)
        true_taxa = list(set(columns)).sort()
        # create combined table
        cm = pd.DataFrame(columns = true_taxa, index = predicted_taxa)
        for c in cm:
            cm[c] = 0
        for i in range(len(df_list)):
            if args.tool == "centrifuge" and files[i].split('-')[3] == 'paired':
                cm = cm.add(df_list[i]*2, fill_value=0)
            else:
                cm = cm.add(df_list[i], fill_value=0)
        cm = cm.fillna(0)
        all_cm[rank] = cm
        print(f'{rank}\t{cm.to_numpy().sum()}')
