import os
import sys
import argparse
import json
import glob
import math
import csv
import datetime
import pandas as pd
import seaborn as sns
import statistics
import matplotlib.pyplot as plt
import numpy as np
import random
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.optim as optim
from models import AlexNet
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, BertConfig
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.calibration import calibration_curve
sys.path.append('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]))
from vis_scripts.testing_utils import *



# def ProcessEmbeddings(args, embeddings):
#     """ Do dimensionality reduction on embeddings """
#     emb_df = pd.DataFrame(embeddings)
#     labels = len(correct_sequence_embeddings)*['Correct'] + len(incorrect_sequence_embeddings)*['Incorrect']
#     # scale the data
#     X = emb_df.values
#     scaler = StandardScaler()
#     emb_scaled_df = scaler.fit_transform(X)
#     # PCA
#     pca = PCA()
#     pca_full = pca.fit(emb_scaled_df)
#     explained_variance_ratio = pca_full.explained_variance_ratio_
#     cumulative_variance = np.cumsum(explained_variance_ratio)
#     emb_transformed = pca.fit_transform(emb_scaled_df)
#     # Plot proportion of the total explained variance for each component and cumulative explained variance
#     plt.figure(figsize=(8, 5))
#     plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='-', label='Explained variance ratio')
#     plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', label='Cumulative explained variance ratio')
#     plt.xlabel('Number of Components')
#     plt.ylabel('Explained Variance Ratio')
#     plt.title('Explained Variance Ratio and Cumulative Explained Variance Ratio vs. Number of Components')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(os.path.join(args.output_dir,'testing', 'pca_variance.png'), dpi=300, bbox_inches='tight')
#     # Visualize the transformed data using a scatter plot
#     pca_result_df = pd.DataFrame({'PCA1': emb_transformed[:, 0], 'PCA2': emb_transformed[:, 1], 'label': labels})
#     fig, ax = plt.subplots(1)
#     sns.scatterplot(x='PCA1', y='PCA2', hue='label', data=pca_result_df, ax=ax,s=120)
#     lim = (emb_transformed.min()-5, emb_transformed.max()+5)
#     ax.set_xlim(lim)
#     ax.set_ylim(lim)
#     ax.set_aspect('equal')
#     ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
#     plt.savefig(os.path.join(args.output_dir,'testing', 'pca_emb_transformed.png'), dpi=300, bbox_inches='tight')
#     print(len(correct_sequence_embeddings))
#     print(len(incorrect_sequence_embeddings))
#     # t-SNE
#     # get TSNE embedding with 2 dimensions
#     n_components = 2
#     tsne = TSNE(n_components)
#     tsne_result = tsne.fit_transform(X)
#     tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': labels})
#     fig, ax = plt.subplots(1)
#     sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120)
#     lim = (tsne_result.min()-5, tsne_result.max()+5)
#     ax.set_xlim(lim)
#     ax.set_ylim(lim)
#     ax.set_aspect('equal')
#     ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
#     plt.savefig(os.path.join(args.output_dir,'testing', 'tsne_emb_transformed.png'), dpi=300, bbox_inches='tight')

def SummarizeResults(args, process, batch_num, sequences_idx, batch_idx, sequences, inputs, batch_predictions, batch_ground_truth, probs, embeddings, attentions, dict_tokens):
    batch_input_ids, _, _, _, _ = inputs
    batch_input_ids = batch_input_ids.tolist()
    genome = ''
    label = ''
    annot_info = {}
    outfile = open(os.path.join(args.output_dir, f'interpretability_info_{batch_num}_{process}.tsv'), 'w')
    for b in range(len(sequences_idx)):
        seq_idx = sequences_idx[b]
        batch_seq_idx = batch_idx[b]
        test_label = sequences[seq_idx][0]
        test_genome = sequences[seq_idx][1]
        # get annotations of testing genome
        if test_genome != genome:
            input_dir = os.getcwd()
            if not os.path.exists(args.annotations_dir):
                os.makedirs(args.annotations_dir)
            test_annot_info, _ = GetAnnotInfo(test_genome, input_dir, args.annotations_dir, args.output_dir)
            correct_genes = defaultdict(list)
            incorrect_genes = defaultdict(list)
            correct_seq = {}
            incorrect_seq = {}
            genome = test_genome
            annot_info = test_annot_info
            if not os.path.exists(os.path.join(args.output_dir, f'label_{test_label}')):
                os.makedirs(os.path.join(args.output_dir, f'label_{test_label}'))

        # get embeddings
        # embeddings shape: (batch_size, 512, 768)
        # hidden_states is a list of tensors, one for each layer and one for the initial embeddings.
        # The last element in the list contains the final layer's hidden states (the contextualized embeddings)
        # if batch in seq_selected and probs[0][batch_predictions[0]] >= args.threshold:
        # verify DNA sequence
        input_ids = batch_input_ids[batch_seq_idx]
        batch_seq = ''
        i = 1
        while i < len(input_ids):
            if input_ids[i] not in [3, 0]:
                if input_ids[i] == 1:
                    u_idx = i
                    while u_idx < len(input_ids):
                        if input_ids[u_idx] == 1:
                            u_idx += 1
                        else:
                            batch_seq += dict_tokens[input_ids[u_idx]]
                            break
                    i = u_idx
                else:
                    if i == 1:
                        batch_seq += dict_tokens[input_ids[i]]
                    else:
                        batch_seq += dict_tokens[input_ids[i]][-1]
            i += 1
        # update original sequence if presence of unknown character
        seq_updated = ''
        i = 0
        while i < len(sequences[seq_idx][2]):
            if sequences[seq_idx][2][i] in ['A','T','C','G']:
                seq_updated += sequences[seq_idx][2][i]
            i += 1
        assert batch_seq == seq_updated, f'not the same sequence: {batch_seq}\t{seq_updated}\t{sequences[seq_idx][2]}'
        result = 'I' if batch_ground_truth[batch_seq_idx] != batch_predictions[batch_seq_idx] else 'C'
        outfile.write(f'{test_label}\t{test_genome}\t{batch_ground_truth[batch_seq_idx]}\t{batch_predictions[batch_seq_idx]}\t{result}\t{probs[batch_seq_idx][batch_predictions[batch_seq_idx]]}\t{len(sequences[seq_idx][2])}\t{sequences[seq_idx][2]}')

        # get embeddings from ['CLS']
        # embeddings = outputs.hidden_states[-1].tolist()
        # write embeddings to file
        outfile.write(f'\t{embeddings[batch_seq_idx][0][0]}')
        for i in range(1, len(embeddings[batch_seq_idx][0]), 1):
            outfile.write(f' {embeddings[batch_seq_idx][0][i]}')
    
        # get attentions
        # Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length)
        # attentions = list(outputs.attentions)
        # get attention scores of the last attention head in the last attention layer for the sequence investigated, shape is (max_position_embeddings, max_position_embeddings)
        # len(attentions) --> 12 attention layers
        # attentions[-1].size() --> torch.Size([1, 12, 512, 512])
        # attentions[-1][0].size() --> torch.Size([12, 512, 512]) --> 12 attention heads
        # attentions[-1][0][-1].size() --> torch.Size([512, 512]) --> last attention head
        # iterate over the scores of the 12 attention layers
        for i in range(len(attentions)):
            # get last attention head ([-1]) of ith attention layer ([i])
            # attentions_scores = attentions[i][batch_seq_idx][-1].tolist()
            attentions_scores = attentions[i][batch_seq_idx][-1]
            df = pd.DataFrame(attentions_scores)
            # get list of tokens
            tokens = [dict_tokens[j] for j in input_ids]
            df.columns = tokens
            # remove rows ['PAD'], ['CLS'] and ['SEP']
            idx_to_rm = [idx for idx in range(len(tokens)) if tokens[idx] in ['[PAD]', '[CLS]', '[SEP]']]
            df = df.drop(idx_to_rm, axis='index')
            # remove columns ['PAD'], ['CLS'] and ['SEP']
            if '[PAD]' in tokens:
                df = df.drop('[PAD]', axis='columns')
            df = df.drop('[CLS]', axis='columns')
            df = df.drop('[SEP]', axis='columns')
            # get list of kmers in the sequence
            df_kmers = df.columns.tolist()
            # rename index to kmers
            df.index = df_kmers
            # save attentions dataframe to file
            df.to_csv(os.path.join(args.output_dir, f'label_{test_label}', f'{test_genome}_attentions_{seq_idx}_{i}.tsv'), sep='\t', index=False)
        # get gene associated with DNA sequence
        seq_start = sequences[seq_idx][3]
        seq_end = sequences[seq_idx][4]
        list_tokens = line.rstrip().split('\t')[1].split(' ')
        seq = list_tokens[0]
        for i in range(len(list_tokens)):
            seq += list_tokens[i][-1]
        gene_id, gene_info = GetGenes(annot_info, seq_start, seq_end)
        gene_info_up = [seq_start, seq_end] + gene_info
        
        if result == 'C':
            correct_genes[gene_id].append(gene_info_up)
            correct_seq[seq_idx] = [seq_start, seq_end]
        elif result == 'I':
            incorrect_genes[gene_id].append(gene_info_up)
            incorrect_seq[seq_idx] = [seq_start, seq_end]

        outfile.write(f'\t{gene_id}')
        if len(gene_info) > 0:
            for i in range(len(gene_info)):
                outfile.write(f'\t{gene_info[i]}')
            if gene_info[0] == 'protein_coding':
                outfile.write('\n')
            else:
                outfile.write('\tNA\tNA\n')
        else:
            for i in range(7):
                outfile.write('\tNA')
            outfile.write('\n')


def train_step(inputs, model, optimizer, device, model_type, batch_size, loss_fn=None):
    if model_type == 'bert':
        input_ids, attention_mask, position_ids, token_type_ids, label = inputs
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        position_ids = position_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        label = label.to(device)
        # set the gradients of tensord to 0
        optimizer.zero_grad()
        # forward
        outputs = model(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=label)
        train_loss = outputs.loss
        # backward + optimize
        train_loss.backward()
        optimizer.step()
        _, predictions = torch.max(outputs.logits, dim=1)
    elif model_type == 'cnn':
        input_ids, label = inputs
        input_ids = input_ids.to(device)
        label = label.to(device)
        # set the gradients of tensord to 0
        optimizer.zero_grad()
        # forward
        outputs = model(input_ids)
        train_loss = loss_fn(outputs, label.squeeze())
        _, predictions = torch.max(outputs, dim=1)
        
    correct = (predictions == torch.flatten(label)).sum().item()
    train_accuracy = correct/batch_size

    return train_loss.item(), train_accuracy

def test_step(inputs, model, device, model_type, batch_size, loss_fn=None):
    if model_type == 'bert':
        input_ids, attention_mask, position_ids, token_type_ids, label = inputs   
        input_ids = input_ids.to(device)
        label = label.to(device)
        attention_mask = attention_mask.to(device)
        position_ids = position_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        outputs = model(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=label, output_hidden_states=True, output_attentions=True)
        logits = outputs.logits
        test_loss = outputs.loss
    elif model_type == 'cnn':
        input_ids, label = inputs
        input_ids = input_ids.to(device)
        label = label.to(device)
        logits = model(input_ids)
        test_loss = loss_fn(logits, label.squeeze())
    
    _, predictions = torch.max(logits, dim=1)
    probs = nn.functional.softmax(logits, dim=1)
    label = torch.flatten(label)
    correct = (predictions == label).sum().item()
    test_accuracy = correct/batch_size

    return test_loss.item(), test_accuracy, predictions.tolist(), label.tolist(), probs.tolist(), logits, input_ids

# class to prepare the input data for training and testing   
class TaxClassDataset(Dataset):
    def __init__(self, tsv_file, tokens_file, label, mode, model_type):
        self.data = pd.read_csv(tsv_file, sep='\t', header=None)
        self.tokens_dict = self.get_tokens_id(tokens_file)
        self.label = label
        self.model_type = model_type
        self.max_position_embedding = 512
        self.mode = mode

    def get_tokens_id(self, tokens_file):
        with open(tokens_file, 'r') as f:
            tokens_dict = {line.rstrip(): idx for idx, line in enumerate(f.readlines())}
        return tokens_dict
    
    def update_label(self, label):
        if label == self.label:
            label = [1]
        else:
            label = [0]
        return label
    
    def prepare_bert_input(self, tokens):
        # adjust the list of tokens according to the max size allowed (max position embedding minus special tokens CLS and SEP)
        if len(tokens) > self.max_position_embedding - 2:
            tokens = tokens[:self.max_position_embedding - 2]
        
        # replace tokens by their id
        input_ids = [self.tokens_dict['[CLS]']]
        for i in range(len(tokens)):
            if tokens[i] in self.tokens_dict:
                input_ids.append(self.tokens_dict[tokens[i]])
            else:
                input_ids.append(self.tokens_dict['[UNK]'])
        input_ids.append(self.tokens_dict['[SEP]'])

        # pad vector if necessary
        if len(input_ids) < self.max_position_embedding:
            num_padded_values = self.max_position_embedding - len(input_ids)
            input_ids = input_ids + [self.tokens_dict['[PAD]']] * num_padded_values
            attention_mask = [1]*(self.max_position_embedding - num_padded_values) + [0] * num_padded_values
        else:
            attention_mask = [1] * self.max_position_embedding

        position_ids = torch.tensor(list(range(self.max_position_embedding)))
        token_type_ids = torch.tensor([0] * self.max_position_embedding)
        return torch.tensor(input_ids), torch.tensor(attention_mask), position_ids, token_type_ids
    
    def prepare_cnn_input(self, tokens):
        # adjust the list of tokens according to the max size allowed (max position embedding minus special tokens CLS and SEP)
        if len(tokens) > self.max_position_embedding - 2:
            tokens = tokens[:self.max_position_embedding - 2]
        
        # replace tokens by their id
        input_ids = []
        for i in range(len(tokens)):
            if tokens[i] in self.tokens_dict:
                input_ids.append(self.tokens_dict[tokens[i]])
            else:
                input_ids.append(self.tokens_dict['[UNK]'])

        # pad vector if necessary
        if len(input_ids) < self.max_position_embedding:
            num_padded_values = self.max_position_embedding - len(input_ids)
            input_ids = input_ids + [self.tokens_dict['[PAD]']] * num_padded_values

        return torch.tensor(input_ids)

    def __len__(self):
        return list(self.data.shape)[0]

    def __getitem__(self, idx):
        label = torch.tensor(self.update_label(self.data.iloc[idx,0]))
        # label = torch.tensor(self.data.iloc[idx,0])
        if self.mode in ['testing','training']:
            tokens = self.data.iloc[idx,1].split(' ') 
        elif self.mode == 'interpretability':
            tokens = self.data.iloc[idx,2].split(' ')
        
        if self.model_type == 'bert':
            input_ids, attention_mask, position_ids, token_type_ids = self.prepare_bert_input(tokens)
            return input_ids, attention_mask, position_ids, token_type_ids, label
        elif self.model_type == 'cnn':
            input_ids = self.prepare_cnn_input(tokens)
            return input_ids, label
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_tsv_file', type=str, help='input file containing labels and reads from training dataset')
    parser.add_argument('--val_tsv_file', type=str, help='input file containing labels and reads from validation dataset')
    parser.add_argument('--tsv_file', type=str, help='path to file with dataset')
    parser.add_argument('--labels', type=str, help='path to file with labels')
    parser.add_argument('--genome', help='do testing at the genome level', action='store_true')
    parser.add_argument('--resume', help='resume training', action='store_true')
    parser.add_argument('--learning_curves', help='create learning curves', action='store_true')
    parser.add_argument('--label', type=int, help='label of interest')
    parser.add_argument('--patience', type=int, help='patience number for early stopping')
    parser.add_argument('--config_file', type=str, help='path to config file containing parameters')
    parser.add_argument('--mode', type=str, help='run script in training or testing mode', choices=['training','testing','interpretability'])
    parser.add_argument('--tokens_file', type=str, help='file with list of tokens')
    parser.add_argument('--kmer', type=str, help='kmer value')
    parser.add_argument('--model', type=str, help='path to model save with Hugging Face function save_pretrained()')
    parser.add_argument('--model_type', type=str, help='type of model', choices=['cnn','bert'])
    parser.add_argument('--batch_size', type=int, help='batch size', default=32)
    parser.add_argument('--num_epochs', type=int, help='number of epochs', default=1)
    parser.add_argument('--sample_size', type=int, help='number of DNA sequences to sample from the test set for embeddings analysis', default=100)
    parser.add_argument('--num_processes', type=int, help='number of processes', default=1)
    parser.add_argument('--threshold', type=float, help='threshold of probability score', default=0.9)
    parser.add_argument('--learning_rate', type=float, help='initial learning rate', default=0.000002)
    parser.add_argument('--taxonomy', type=str, help='path to file mapping labels to taxonomy')
    parser.add_argument('--output_dir', type=str, help='path to output directory', default=os.getcwd())
    parser.add_argument('--lc_dir', type=str, help='input directory for creating learning curves')
    parser.add_argument('--testing_dir', type=str, help='input directory for summarizing testing results')
    parser.add_argument('--annotations_dir', type=str, help='path to directory to store annotations downloaded from NCBI')
    parser.add_argument('--epoch_to_resume', type=int, help='epoch to resume from for cnn model')
    args = parser.parse_args()

    start = datetime.datetime.now()

    # allow usage of GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # create output directory
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    if args.mode == "training":

        if not os.path.isdir(os.path.join(args.output_dir, 'logs')):
            os.makedirs(os.path.join(args.output_dir, 'logs'))

        if not os.path.isdir(os.path.join(args.output_dir, 'model')):
            os.makedirs(os.path.join(args.output_dir, 'model'))

        # prepare input data
        train_data = TaxClassDataset(args.train_tsv_file, args.tokens_file, args.label, args.mode, args.model_type)
        val_data = TaxClassDataset(args.val_tsv_file, args.tokens_file, args.label, args.mode, args.model_type)
        train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)
        
        # load parameters
        with open(args.config_file, "r") as f:
            config_dict = json.load(f)
        print(config_dict)
        if args.model_type == 'bert':
            # create BERT config object and model
            bert_config = BertConfig(vocab_size=config_dict["vocab_size"])
            if args.resume:
                model = BertForSequenceClassification.from_pretrained(args.model, config=bert_config)
            else:
                model = BertForSequenceClassification(config=bert_config)
            model.to(device)
            embeddings = model.bert.embeddings.word_embeddings.weight
        elif args.model_type == 'cnn':
            if args.resume:
                # load model in SavedModel format
                #model = tf.keras.models.load_model(args.model)
                # load model saved with checkpoints
                model = AlexNet(config_dict["vector_size"], config_dict["embedding_size"], config_dict["num_classes"], config_dict["vocab_size"], config_dict["dropout_rate"])
                checkpoint = tf.train.Checkpoint(optimizer=opt, model=model)
                checkpoint.restore(os.path.join(args.ckpt, f'ckpt-{args.epoch_to_resume}')).expect_partial()
            else:
                model = AlexNet(config_dict["vector_size"], config_dict["embedding_size"], config_dict["num_classes"], config_dict["vocab_size"], config_dict["dropout_rate"])
            model.to(device)
            embeddings = model.embedding.weight.detach().cpu().numpy()
            loss_fn = nn.CrossEntropyLoss()
            print(model)
        
        data = []
        with open(args.tokens_file, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                token_embeddings = embeddings[idx].tolist()
                token_embeddings.insert(0,line.rstrip())
                data.append(token_embeddings)

        with open(os.path.join(args.output_dir, 'token_embeddings_initial.csv'), mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)

        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        train_logs_file = open(os.path.join(args.output_dir, 'logs', 'training.tsv'), 'w')
        val_logs_file = open(os.path.join(args.output_dir, 'logs', 'validation.tsv'), 'w')

        with open(args.train_tsv_file, 'r') as f:
            num_train_reads = len(f.readlines())

        with open(args.val_tsv_file, 'r') as f:
            num_val_reads = len(f.readlines())

        print(f'num_train_reads\t{num_train_reads}\nnum_val_reads\t{num_val_reads}\n'
            f'train_steps\t{math.ceil((num_train_reads/args.batch_size)*args.num_epochs)}\nval_steps\t{math.ceil((num_val_reads/args.batch_size)*args.num_epochs)}\n')

        # define variables for early stopping
        best_val_accuracy = np.Inf
        patience = args.patience
        wait = 0
        best_model = None
        best_loss = np.Inf
        stop_training = False
        found_min = False
        min_epoch = 0

        for epoch in range(args.num_epochs):
            print('EPOCH', epoch)
            epoch_train_loss = 0.0
            epoch_train_acc = 0.0
            for train_batch, inputs in enumerate(train_dataloader, 0):
                if args.model_type == 'bert':
                    train_loss, train_accuracy = train_step(inputs, model, optimizer, device, args.model_type, args.batch_size)
                elif args.model_type == 'cnn':
                    train_loss, train_accuracy = train_step(inputs, model, optimizer, device, args.model_type, args.batch_size, loss_fn)
                epoch_train_loss += train_loss
                epoch_train_acc += train_accuracy
                if (train_batch+1) % 100 == 0:
                    print(f'epoch: {epoch+1}\tbatch: {train_batch+1}\ttraining loss: {round(epoch_train_loss/(train_batch+1),3)}\ttraining accuracy: {round(epoch_train_acc/(train_batch+1),3)*100}')
                train_logs_file.write(f'{epoch+1}\t{train_batch+1}\t{round(epoch_train_loss/(train_batch+1),3)}\t{round(epoch_train_acc/(train_batch+1),3)*100}\t{optimizer.param_groups[0]['lr']}\n')

            epoch_val_loss = 0.0
            epoch_val_acc = 0.0
            for val_batch, inputs in enumerate(val_dataloader, 0):
                if args.model_type == 'bert':
                    val_loss, val_accuracy, _, _, _, _ = test_step(inputs, model, device, args.model_type, args.batch_size)
                elif args.model_type == 'cnn':
                    val_loss, val_accuracy, pred, labels, probs, logits, input_ids = test_step(inputs, model, device, args.model_type, args.batch_size, loss_fn)
                    # print('loss', val_loss)
                    # print('accuracy', val_accuracy)
                    # print('pred', pred)
                    # print('label', labels)
                    # print('prob', probs)
                    # print('logits', logits)
                    # print('input_ids', input_ids)
                epoch_val_loss += val_loss
                epoch_val_acc += val_accuracy
            epoch_val_loss = round(epoch_val_loss/(val_batch+1),3)
            epoch_val_acc = round(epoch_val_acc/(val_batch+1),3)
            val_logs_file.write(f'{epoch+1}\t{val_batch+1}\t{epoch_val_loss}\t{epoch_val_acc*100}\t{optimizer.param_groups[0]['lr']}\t{wait}\n')

            # check validation loss at the end of epoch
            # if patience == args.patience:
            # if wait >= patience:
            #     lr = optimizer.param_groups[0]['lr']
            #     if (lr == args.learning_rate) and (args.learning_rate != 0.000002):
            #         optimizer.param_groups[0]['lr'] = 0.000002
            #         # patience = 0
            #     else:
            #         stop_training = True
            # else:
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                best_val_accuracy = epoch_val_acc
                best_model = model.state_dict()
                # patience = 0 # Reset wait counter
                wait = 0
                min_epoch = epoch + 1
                found_min = True
            else:
                wait += 1
                if wait >= patience:
                    # lower learning rate
                    lr = optimizer.param_groups[0]['lr']
                    if (lr == args.learning_rate) and (args.learning_rate != 0.000002):
                        optimizer.param_groups[0]['lr'] = 0.000002
                        wait = 0
                        # patience = 0
                    else:
                        print(f"Early stopping at epoch {epoch+1}")
                        stop_training = True
                # patience += 1
            print(f'epoch: {epoch+1}\tval batch: {val_batch+1}\tvalidation loss: {epoch_val_loss}\tvalidation accuracy: {epoch_val_acc*100}\t{wait}')
            
            # save best model obtained so far
            if found_min and stop_training == False:
                if args.model_type == 'bert':
                    model.save_pretrained(os.path.join(args.output_dir, 'model', f'model-epoch-{epoch+1}'))
                torch.save(model.state_dict(), os.path.join(args.output_dir, 'model', f'model-epoch-{epoch+1}.pth'))

            if stop_training or (epoch+1) == args.num_epochs:
                if found_min:
                    # save best model
                    torch.save(best_model, os.path.join(args.output_dir, 'model', f'model-epoch-{min_epoch}-best.pth'))
                    model.load_state_dict(best_model)
                    if args.model_type == 'bert':
                        model.save_pretrained(os.path.join(args.output_dir, 'model', f'model-epoch-{min_epoch}-best'))
                else:
                    if args.model_type == 'bert':
                        # save model if training has reached the max number of epochs 
                        model.save_pretrained(os.path.join(args.output_dir, 'model', f'model-epoch-{epoch+1}'))
                    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model', f'model-epoch-{epoch+1}.pth'))
                break

        train_logs_file.close()
        val_logs_file.close()

        end = datetime.datetime.now()
        total_time = end - start
        hours, seconds = divmod(total_time.seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        days = total_time.days
        
        if args.model_type == 'bert':
            embeddings = model.bert.embeddings.word_embeddings.weight
        elif args.model_type == 'cnn':
            embeddings = model.embedding.weight.detach().cpu().numpy()
        data = []
        with open(args.tokens_file, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                token_embeddings = embeddings[idx].tolist()
                token_embeddings.insert(0,line.rstrip())
                data.append(token_embeddings)

        with open(os.path.join(args.output_dir, 'token_embeddings_final.csv'), mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(data)

        with open(os.path.join(args.output_dir, f'{args.mode}_summary.tsv'), 'w') as f:
            f.write(f'Runtime\t{days}:{hours}:{minutes}:{seconds}:{total_time.microseconds}\n')


    if args.mode == "interpretability":
        
        # prepare input data
        data = TaxClassDataset(args.tsv_file, args.tokens_file, args.label, args.mode)
        # dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)
        dataloader = DataLoader(data, batch_size=1, shuffle=False)
        
        # load parameters for BERT
        with open(args.bert_config_file, "r") as f:
            config_dict = json.load(f)
        print(config_dict)
        
        # create BERT config object and model
        bert_config = BertConfig(vocab_size=config_dict["vocab_size"])
        model = BertForSequenceClassification.from_pretrained(args.model, config=bert_config)
        model.to(device)

        start = datetime.datetime.now()

        # with open(args.tsv_file, 'r') as f:
        #     num_test_reads = len(f.readlines())

        dict_tokens = {}
        with open(args.tokens_file, 'r') as f:
            for idx, line in enumerate(f):
                dict_tokens[idx] = line.rstrip()
        
        # load DNA sequences
        sequences_info = []
        with open(args.tsv_file, 'r') as f:
            for idx, line in enumerate(f):
                label = line.rstrip().split('\t')[0]
                genome_id = line.rstrip().split('\t')[1]
                list_tokens = line.rstrip().split('\t')[2].split(' ')
                seq = list_tokens[0]
                for i in range(1, len(list_tokens), 1):
                    seq += list_tokens[i][-1]
                sequences_info.append([label, genome_id, seq, int(line.rstrip().split('\t')[3]), int(line.rstrip().split('\t')[4])])
        print(f'# sequences: {len(sequences_info)}\n{sequences_info[:5]}')
        
        # prev_batch_size = 0
        genome = ''
        label = ''
        annot_info = {}
        outfile = open(os.path.join(args.output_dir, f'interpretability_info.tsv'), 'w')
        for batch, inputs in enumerate(dataloader, 0):
            _, _, batch_predictions, batch_ground_truth, probs, outputs = test_step(inputs, model, device, 'test')
            print(f'batch size: {len(batch_predictions)}')
            print(f'batch: {batch}')
            # sequences_idx = [i for i in range(batch*prev_batch_size,(batch*prev_batch_size)+len(batch_predictions),1)]
            # print(sequences_idx)
            # chunk_size = math.ceil(len(sequences_idx)/args.num_processes)
            # grouped_sequences_idx = [sequences_idx[i:i+chunk_size] for i in range(0, len(sequences_idx), chunk_size)]
            # batch_seq_idx = list(range(len(batch_predictions)))
            # grouped_sequences_batch_idx = [batch_seq_idx[i:i+chunk_size] for i in range(0, len(batch_seq_idx), chunk_size)]
            # prev_batch_size = len(batch_predictions)
            # embeddings = outputs.hidden_states[-1].tolist()
            # print('start attention conversion', datetime.datetime.now())
            # attentions = list(outputs.attentions)
            # attentions = [i.tolist() for i in attentions]
            # print('end attention conversion', datetime.datetime.now())
            # with mp.Manager() as manager: # create manager object to allow processes to manipulate python data structures
            #     # create list of Process objects
            #     processes = [mp.Process(target=SummarizeResults, args=(args, i, batch, grouped_sequences_idx[i], grouped_sequences_batch_idx[i], sequences_info, inputs, batch_predictions, batch_ground_truth, probs, embeddings, attentions, dict_tokens)) for i in range(len(grouped_sequences_idx))]
            #     for p in processes:
            #         p.start()
            #     for p in processes:
            #         p.join()   
               
            test_label = sequences_info[batch][0]
            test_genome = sequences_info[batch][1]
            # get annotations of testing genome
            if test_genome != genome:
                input_dir = os.getcwd()
                if not os.path.exists(args.annotations_dir):
                    os.makedirs(args.annotations_dir)
                test_annot_info, _ = GetAnnotInfo(test_genome, input_dir, args.annotations_dir, args.output_dir)
                correct_genes = defaultdict(list)
                incorrect_genes = defaultdict(list)
                correct_seq = {}
                incorrect_seq = {}
                genome = test_genome
                annot_info = test_annot_info
                if not os.path.exists(os.path.join(args.output_dir, f'label_{test_label}')):
                    os.makedirs(os.path.join(args.output_dir, f'label_{test_label}'))

            # get embeddings
            # embeddings shape: (batch_size, 512, 768)
            # hidden_states is a list of tensors, one for each layer and one for the initial embeddings.
            # The last element in the list contains the final layer's hidden states (the contextualized embeddings)
            # if batch in seq_selected and probs[0][batch_predictions[0]] >= args.threshold:
            # verify DNA sequence
            input_ids, _, _, _, _ = inputs
            input_ids = input_ids.tolist()[0]
            batch_seq = ''
            i = 1
            while i < len(input_ids):
                if input_ids[i] not in [3, 0]:
                    if input_ids[i] == 1:
                        u_idx = i
                        while u_idx < len(input_ids):
                            if input_ids[u_idx] == 1:
                                u_idx += 1
                            else:
                                batch_seq += dict_tokens[input_ids[u_idx]]
                                break
                        i = u_idx
                    else:
                        if i == 1:
                            batch_seq += dict_tokens[input_ids[i]]
                        else:
                            batch_seq += dict_tokens[input_ids[i]][-1]
                i += 1
            # update original sequence if presence of unknown character
            seq_updated = ''
            i = 0
            while i < len(sequences_info[batch][2]):
                if sequences_info[batch][2][i] in ['A','T','C','G']:
                    seq_updated += sequences_info[batch][2][i]
                i += 1
            assert batch_seq == seq_updated, f'not the same sequence: {batch_seq}\t{seq_updated}\t{sequences_info[batch][2]}'
            result = 'I' if batch_ground_truth[0] != batch_predictions[0] else 'C'
            outfile.write(f'{test_label}\t{test_genome}\t{batch_ground_truth[0]}\t{batch_predictions[0]}\t{result}\t{probs[0][batch_predictions[0]]}\t{len(sequences_info[batch][2])}\t{sequences_info[batch][2]}')

            # get embeddings from ['CLS']
            embeddings = outputs.hidden_states[-1].tolist()
            # write embeddings to file
            outfile.write(f'\t{embeddings[0][0][0]}')
            for i in range(1, len(embeddings[0][0]), 1):
                outfile.write(f' {embeddings[0][0][i]}')
    
            # get attentions
            # Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length)
            attentions = list(outputs.attentions)
            # get attention scores of the last attention head in the last attention layer for the sequence investigated, shape is (max_position_embeddings, max_position_embeddings)
            # len(attentions) --> 12 attention layers
            # attentions[-1].size() --> torch.Size([1, 12, 512, 512])
            # attentions[-1][0].size() --> torch.Size([12, 512, 512]) --> 12 attention heads
            # attentions[-1][0][-1].size() --> torch.Size([512, 512]) --> last attention head
            # iterate over the scores of the 12 attention layers
            for i in range(len(attentions)):
                # get last attention head ([-1]) of ith attention layer ([i])
                attentions_scores = attentions[i][0][-1].tolist()
                df = pd.DataFrame(attentions_scores)
                # get list of tokens
                tokens = [dict_tokens[j] for j in input_ids]
                df.columns = tokens
                # remove rows ['PAD'], ['CLS'] and ['SEP']
                idx_to_rm = [idx for idx in range(len(tokens)) if tokens[idx] in ['[PAD]', '[CLS]', '[SEP]']]
                df = df.drop(idx_to_rm, axis='index')
                # remove columns ['PAD'], ['CLS'] and ['SEP']
                if '[PAD]' in tokens:
                    df = df.drop('[PAD]', axis='columns')
                df = df.drop('[CLS]', axis='columns')
                df = df.drop('[SEP]', axis='columns')
                # get list of kmers in the sequence
                df_kmers = df.columns.tolist()
                # rename index to kmers
                df.index = df_kmers
                # save attentions dataframe to file
                df.to_csv(os.path.join(args.output_dir, f'label_{test_label}', f'{test_genome}_attentions_{batch}_{i}.tsv'), sep='\t', index=False)
            # get gene associated with DNA sequence
            seq_start = sequences_info[batch][3]
            seq_end = sequences_info[batch][4]
            list_tokens = line.rstrip().split('\t')[1].split(' ')
            seq = list_tokens[0]
            for i in range(len(list_tokens)):
                seq += list_tokens[i][-1]
            gene_id, gene_info = GetGenes(annot_info, seq_start, seq_end)
            gene_info_up = [seq_start, seq_end] + gene_info
            
            if result == 'C':
                correct_genes[gene_id].append(gene_info_up)
                correct_seq[batch] = [seq_start, seq_end]
            elif result == 'I':
                incorrect_genes[gene_id].append(gene_info_up)
                incorrect_seq[batch] = [seq_start, seq_end]

            outfile.write(f'\t{gene_id}')
            if len(gene_info) > 0:
                for i in range(len(gene_info)):
                    outfile.write(f'\t{gene_info[i]}')
                if gene_info[0] == 'protein_coding':
                    outfile.write('\n')
                else:
                    outfile.write('\tNA\tNA\n')
            else:
                for i in range(7):
                    outfile.write('\tNA')
                outfile.write('\n')


        # # visualize incorrect and correct classifications on circos plot 
        # if args.genome:
        #     CircosPlot(correct_seq, incorrect_seq, correct_genes, incorrect_genes, train_fasta, test_fasta, test_genome_id, output_dir, args.num_processes)

        end = datetime.datetime.now()
        total_time = end - start
        hours, seconds = divmod(total_time.seconds, 3600)
        minutes, seconds = divmod(seconds, 60)

        with open(os.path.join(args.output_dir, f'{args.mode}_runtime.tsv'), 'w') as f:
            f.write(f'Runtime\t{hours}:{minutes}:{seconds}:{total_time.microseconds}\n')

    if args.mode == "testing":
        # prepare input data
        test_data = TaxClassDataset(args.tsv_file, args.tokens_file, args.label, args.mode)
        test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
        
        # load parameters for BERT
        with open(args.bert_config_file, "r") as f:
            config_dict = json.load(f)
        print(config_dict)
        
        # create BERT config object and model
        bert_config = BertConfig(vocab_size=config_dict["vocab_size"])
        model = BertForSequenceClassification.from_pretrained(args.model, config=bert_config)
        model.to(device)

        start = datetime.datetime.now()

        test_metrics = open(os.path.join(args.output_dir, 'metrics.tsv'), 'w')
        test_sum = open(os.path.join(args.output_dir, 'summary.tsv'), 'w')

        with open(args.tsv_file, 'r') as f:
            num_test_reads = len(f.readlines())

        dict_tokens = {}
        with open(args.tokens_file, 'r') as f:
            for idx, line in enumerate(f):
                dict_tokens[idx] = line.rstrip()
        
        # load DNA sequences
        test_sequences = []
        with open(args.tsv_file, 'r') as f:
            for idx, line in enumerate(f):
                list_tokens = line.rstrip().split('\t')[1].split(' ')
                seq = list_tokens[0]
                for i in range(1, len(list_tokens), 1):
                    seq += list_tokens[i][-1]
                test_sequences.append([seq, int(line.rstrip().split('\t')[2]), int(line.rstrip().split('\t')[3]), int(line.rstrip().split('\t')[4])])

        epoch_test_loss = 0.0
        epoch_test_acc = 0.0
        ground_truth = []
        predictions = []
        confidence_scores = []
        # all_pct_id_p_genome = []
        # all_pct_id_n_genome = []
        # # randomly select sequences for analysis of embeddings 
        # seq_selected = random.sample(range(0, len(test_sequences) + 1), args.sample_size)
        # print(f'# sequences: {len(seq_selected)}')
        # correct_sequence_embeddings = []
        # incorrect_sequence_embeddings = []
        
        for batch, inputs in enumerate(test_dataloader, 0):
            input_ids, attention_mask, position_ids, token_type_ids, label = inputs
            test_loss, test_accuracy, batch_predictions, batch_ground_truth, probs, outputs = test_step(inputs, model, device, 'test')
            epoch_test_loss += test_loss
            epoch_test_acc += test_accuracy
            ground_truth += batch_ground_truth
            predictions += batch_predictions
            confidence_scores += probs
            # _, _, _, _, _, pct_id_p_genome, pct_id_n_genome = inputs
            # all_pct_id_p_genome += pct_id_p_genome.tolist()
            # all_pct_id_n_genome += pct_id_n_genome.tolist()
            
        # update testing loss
        epoch_test_loss = round(epoch_test_loss/(batch+1),3)
        
        assert len(predictions) == len(ground_truth) == num_test_reads, f'problem with vectors: predictions: {len(predictions)}\tground truth: {len(ground_truth)}'

        # get number of FP, FN, TP, TN
        FP_cs = []
        FN_cs = []
        TN_cs = []
        TP_cs = []
        # FP_pct_p = []
        # FN_pct_p = []
        # TN_pct_p = []
        # TP_pct_p = []
        # FP_pct_n = []
        # FN_pct_n = []
        # TN_pct_n = []
        # TP_pct_n = []
        with open(os.path.join(args.output_dir, 'cs_length.tsv'), 'w') as f:
            for i in range(len(predictions)):
                f.write(f'{test_sequences[i][3]}\t')
                if ground_truth[i] == 1 and predictions[i] == 1:
                    TP_cs.append(confidence_scores[i][1])
                    f.write(f'{confidence_scores[i][1]}\tTP\n')
                    # TP_pct_p.append(all_pct_id_p_genome[i])
                    # TP_pct_n.append(all_pct_id_n_genome[i])
                elif ground_truth[i] == 1 and predictions[i] == 0:
                    FN_cs.append(confidence_scores[i][0])
                    f.write(f'{confidence_scores[i][0]}\tFN\n')
                    # FN_pct_p.append(all_pct_id_p_genome[i])
                    # FN_pct_n.append(all_pct_id_n_genome[i])
                elif ground_truth[i] == 0 and predictions[i] == 0:
                    TN_cs.append(confidence_scores[i][0])
                    f.write(f'{confidence_scores[i][0]}\tTN\n')
                    # TN_pct_p.append(all_pct_id_p_genome[i])
                    # TN_pct_n.append(all_pct_id_n_genome[i])
                elif ground_truth[i] == 0 and predictions[i] == 1:
                    FP_cs.append(confidence_scores[i][1])
                    f.write(f'{confidence_scores[i][1]}\tFP\n')
                    # FP_pct_p.append(all_pct_id_p_genome[i])
                    # FP_pct_n.append(all_pct_id_n_genome[i])
        accuracy = round((len(TP_cs)+len(TN_cs))/(len(TP_cs)+len(TN_cs)+len(FN_cs)+len(FP_cs)),3)
        print(accuracy, epoch_test_acc)
        test_sum.write(f'accuracy\t{accuracy}\nloss\t{epoch_test_loss}\n#examples\t{len(predictions)}\n')
        test_sum.write(f'TP\t{len(TP_cs)}\nFN\t{len(FN_cs)}\nTN\t{len(TN_cs)}\nFP\t{len(FP_cs)}\n')
        
        try:
            pos_precision = round(len(TP_cs)/(len(TP_cs)+len(FP_cs)),3)
        except ZeroDivisionError:
            pos_precision = 0
        test_metrics.write(f'1\tprecision\t{pos_precision}\n')

        try:
            neg_precision = round(len(TN_cs)/(len(TN_cs)+len(FN_cs)),3)
        except ZeroDivisionError:
            neg_precision = 0
        test_metrics.write(f'0\tprecision\t{neg_precision}\n')

        try:
            pos_recall = round(len(TP_cs)/(len(TP_cs)+len(FN_cs)),3)
        except ZeroDivisionError:
            pos_recall = 0
        test_metrics.write(f'1\trecall\t{pos_recall}\n')

        try:
            neg_recall = round(len(TN_cs)/(len(TN_cs)+len(FP_cs)),3)
        except ZeroDivisionError:
            neg_recall = 0
        test_metrics.write(f'0\trecall\t{neg_recall}\n')

        

        
        # # plot distribution of percent identity with positive training genome
        # print('Positive', len(TP_pct_p), len(FN_pct_p), len(TN_pct_p), len(FP_pct_p))
        # print('Negative', len(TP_pct_n), len(FN_pct_n), len(TN_pct_n), len(FP_pct_n))
        # values_p = TP_pct_p + FN_pct_p + TN_pct_p + FP_pct_p
        # values_n = TP_pct_n + FN_pct_n + TN_pct_n + FP_pct_n
        # groups_p = ['TP']*len(TP_pct_p)+['FN']*len(FN_pct_p)+['TN']*len(TN_pct_p)+['FP']*len(FP_pct_p)
        # groups_n = ['TP']*len(TP_pct_n)+['FN']*len(FN_pct_n)+['TN']*len(TN_pct_n)+['FP']*len(FP_pct_n)
        # genomes_value = ['Positive']*len(values_p) + ['Negative']*len(values_n)
        # data = {'value': values_p + values_n, 'group': groups_p + groups_n, 'genome': genomes_value}
        # df = pd.DataFrame(data)
        # plot = sns.FacetGrid(df, row='group', col='genome', sharey=False)
        # plot.map_dataframe(sns.histplot, data=data)
        # plt.savefig(os.path.join(args.output_dir, 'pct_identity.png'), dpi=300)
        # plt.clf()
        # # plot distribution of confidence scores per group
        # if len(TP_cs) > 0:
        #     print('TP-CS', min(TP_cs), max(TP_cs), statistics.mean(TP_cs))
        # if len(TN_cs) > 0:
        #     print('TN-CS', min(TN_cs), max(TN_cs), statistics.mean(TN_cs))
        # if len(FP_cs) > 0:
        #     print('FP-CS', min(FP_cs), max(FP_cs), statistics.mean(FP_cs))
        # if len(FN_cs) > 0:
        #     print('FN-CS', min(FN_cs), max(FN_cs), statistics.mean(FN_cs))
        # print('prepare cs plot')
        # sns.histplot(FP_cs, color='pink', label = 'FP')
        # sns.histplot(TP_cs, color='skyblue', label = 'TP')
        # leg = plt.legend(loc = 'upper left')
        # plt.xlabel('Confidence score')
        # plt.ylabel('Frequency')
        # plt.title('Positive class')
        # plt.savefig(os.path.join(args.output_dir, 'confidence_scores_pos.png'), dpi=300)
        # plt.clf()
        # sns.histplot(FN_cs, color='pink', label = 'FN')
        # sns.histplot(TN_cs, color='skyblue', label = 'TN')
        # leg = plt.legend(loc = 'upper left')
        # plt.xlabel('Confidence score')
        # plt.ylabel('Frequency')
        # plt.title('Negative class')
        # plt.savefig(os.path.join(args.output_dir, 'confidence_scores_neg.png'), dpi=300)
        # plt.clf()
        # # Measure overconfidence
        # # calibration curve with sklearn
        # n_bins = 10
        # true_cs = [i[1] for i in confidence_scores]
        # prob_true, prob_pred = calibration_curve(ground_truth, true_cs, n_bins=n_bins, strategy='uniform', pos_label=1)
        # # prob_true = proportion of samples in each bin whose class is the positive class
        # # prob_pred = mean predicted probability for the positive class in each bin.
        # print('prob_true', prob_true)
        # print('prob_pred', prob_pred)
        # # expected calibration error
        # bin_counts = np.histogram(true_cs, bins=n_bins, range=(0, 1))[0]
        # bin_weights = bin_counts / len(true_cs)
        # nonzero = bin_counts > 0
        # print(bin_counts)
        # print(bin_weights)
        # print(nonzero)
        # ece = np.sum(np.abs(prob_true[nonzero] - prob_pred[nonzero]) * bin_weights[nonzero])
        # test_sum.write(f'ECE\t{ece}')    
        # plt.plot(prob_pred, prob_true, marker='.', label = 'BERT')
        # plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Ideally Calibrated')
        # leg = plt.legend(loc = 'upper left')
        # plt.xlabel('Average Predicted Probability in each bin')
        # plt.ylabel('Ratio of positives')
        # plt.savefig(os.path.join(args.output_dir, 'calibration_curve_sklearn.png'), dpi=300)
        
        test_metrics.close()
        test_sum.close()

        end = datetime.datetime.now()
        total_time = end - start
        hours, seconds = divmod(total_time.seconds, 3600)
        minutes, seconds = divmod(seconds, 60)

        with open(os.path.join(args.output_dir, f'{args.mode}_runtime.tsv'), 'w') as f:
            f.write(f'Runtime\t{hours}:{minutes}:{seconds}:{total_time.microseconds}\n')

    if args.lc_dir is not None:
        # create learning curves
        # get input data
        training_file = os.path.join(args.lc_dir, 'logs/training.tsv')
        validation_file = os.path.join(args.lc_dir, 'logs/validation.tsv')
        best_epoch = glob.glob(os.path.join(args.lc_dir, 'model/*-best'))[0]
        best_epoch = int(best_epoch.split('-')[-2]) + 1
        print('best epoch', best_epoch)
        # batch_size = []
        # values = []
        # metric = []
        # dataset = []
        # epochs = []
        # for i in range(len(training_files)):
            # train_bs = training_files[i].split('/')[-3].split('-')[-1]
            # val_bs = training_files[i].split('/')[-3].split('-')[-1]
            # assert train_bs == val_bs
        train_df = pd.read_csv(training_file, sep='\t', header=None)
        val_df = pd.read_csv(validation_file, sep='\t', header=None)
        num_epochs, _ = val_df.shape
        num_steps_per_epoch = train_df.iloc[:, 1].tolist()[-1]
        print(num_steps_per_epoch)
        train_accuracy = [train_df.iloc[:, 3].tolist()[j] for j in range(0, len(train_df), num_steps_per_epoch)]
        train_loss = [train_df.iloc[:, 2].tolist()[j] for j in range(0, len(train_df), num_steps_per_epoch)]
        # values += train_accuracy
        # values += train_loss
        # dataset += ['training']*(len(train_accuracy)*2)
        # metric += ['accuracy']*len(train_accuracy) + ['loss']*len(train_loss)
        val_accuracy = val_df.iloc[:, 3].tolist()
        val_loss = val_df.iloc[:, 2].tolist()
        # values += val_accuracy
        # values += val_loss
        # batch_size += [int(train_bs)]*(len(train_accuracy)*2+len(val_accuracy)*2)
        # dataset += ['validation']*(len(val_accuracy)*2)
        # metric += ['accuracy']*len(val_accuracy) + ['loss']*len(val_loss)
        # epochs += list(range(1, num_epochs+1, 1))*4
        # create dataframe
        values = train_accuracy + val_accuracy + train_loss + val_loss
        metric = ['accuracy']*(len(train_accuracy)+len(val_accuracy)) + ['loss']*(len(train_loss)+len(val_loss))
        dataset = ['training']*len(train_accuracy) + ['validation']*len(val_accuracy) + ['training']*len(train_loss) + ['validation']*len(val_loss)
        epoch = 4*list(range(1,len(train_accuracy)+1,1))
        assert len(values) == len(epoch) == len(dataset) == len(metric)
        data = {'value': values, 'metric': metric, 'dataset': dataset, 'epoch': epoch}
        df = pd.DataFrame(data)
        max_loss = max(df.loc[df['metric'] == 'loss', 'value'].tolist())
        min_loss = min(df.loc[df['metric'] == 'loss', 'value'].tolist())
        print(df)
        print(df.shape)
        print(max_loss, min_loss)
        palette = {'training': 'black', 'validation': 'red'}
        plot = sns.FacetGrid(df, row=None, col='metric', sharey=False)
        plot.map_dataframe(sns.lineplot, x='epoch', y='value', data=data, hue='dataset', palette=palette)
        axes = plot.axes.flatten()
        axes_title = ['','']
        axes_y_labels = ['Accuracy', 'Loss']
        axes_x_labels = ['Epoch', 'Epoch']
        for idx, ax in enumerate(axes):
            ax.set_title(axes_title[idx])
            ax.set_ylabel(axes_y_labels[idx])
            ax.set_xlabel(axes_x_labels[idx])
            ax.lines[0].set_color('black')
            ax.lines[0].set_linestyle('-')
            ax.lines[1].set_color('red')
            ax.lines[1].set_linestyle('-')
            # add vertical line to define best checkpoint
            ax.axvline(x=best_epoch, color='blue', linestyle='--', linewidth=2)
            if idx == 0:
                ax.set_ylim(0,100)
            if idx == 1:
                ax.set_ylim(min_loss,max_loss)
            print(idx, ax.get_title(), ax.get_ylabel(), ax.get_xlabel(), ax.get_ylim())
        plot.add_legend(loc='lower center')
        plt.savefig(os.path.join(args.lc_dir, 'logs', 'learning_curves.png'), dpi=300)


    # create plots of accuracy,precision and recall for multiple models
    if args.testing_dir is not None:
        with open(args.labels, 'r') as f:
            list_labels = [line.rstrip() for line in f.readlines()]  
        print(list_labels)
        # load data
        values = []
        metrics = []
        labels = []
        for l in list_labels:
            print(l)
            # get precision and recall for labels 0 and 1
            metrics_file = os.path.join(args.testing_dir, f'label_{l}/torch/k4/species_dataset_1_patience_10/testing/dataset/metrics.tsv')
            metrics_df =  pd.read_csv(metrics_file, sep='\t', header=None)
            metrics_df.columns = ['label','metric','value']
            # print(metrics_df)
            values += metrics_df['value'].tolist()
            metrics += metrics_df['metric'].tolist()
            labels += metrics_df['label'].tolist()
            # get accuracy
            summary_file = os.path.join(args.testing_dir, f'label_{l}/torch/k4/species_dataset_1_patience_10/testing/dataset/summary.tsv')
            summary_df =  pd.read_csv(summary_file, sep='\t', header=None)
            summary_df.columns = ['metric','value']
            # print(summary_df)
            values += [summary_df['value'].tolist()[0]]
            metrics += ['accuracy']
        print(values)
        print(metrics)
        print(labels)
        data = {'value': values, 'metric': metrics, 'label':labels}
        df = pd.DataFrame(data)
        print(df)
        df['label'].replace(0, 'label 0', inplace=True)
        df['label'].replace(1, 'label 1', inplace=True)
        print(df)
        # palette = {'label 0': 'orange', 'label 1': 'pink'}
        # plot = sns.FacetGrid(df, row=None, col='metric', sharey=False)
        # plot.map_dataframe(sns.lineplot, x='epoch', y='value', data=data, hue='label', palette=palette)
        # axes = plot.axes.flatten()
        # axes_title = ['','']
        # axes_y_labels = ['Precision', 'Recall']
        # axes_x_labels = ['Epoch', 'Epoch']
        # for idx, ax in enumerate(axes):
        #     ax.set_title(axes_title[idx])
        #     ax.set_ylabel(axes_y_labels[idx])
        #     ax.set_xlabel(axes_x_labels[idx])
        #     # ax.lines[0].set_color('black')
        #     # ax.lines[0].set_linestyle('-')
        #     # ax.lines[1].set_color('red')
        #     # ax.lines[1].set_linestyle('-')
        #     # add vertical line to define best checkpoint
        #     ax.axvline(x=best_epoch, color='blue', linestyle='--', linewidth=2)
        #     if idx == 0:
        #         ax.set_ylim(0,1)
        #     if idx == 1:
        #         ax.set_ylim(0,1)
        # plot.add_legend(loc='lower center')
        # plt.savefig(os.path.join(args.testing_dir, 'testing/dataset', 'metrics.png'), dpi=300)

        # # plot accuracy
        # values = []
        # epochs = []
        # for i in range(len(summary_file)):
        #     if len(metrics_file[i].split('/')[-2].split('-')) > 1:
        #         epoch = int(metrics_file[i].split('/')[-2].split('-')[0])
        #     else:
        #         epoch = int(metrics_file[i].split('/')[-2])
        #     summary_df =  pd.read_csv(summary_file[i], sep='\t', header=None)
        #     summary_df.columns = ['metric','value']
        #     values += [summary_df['value'].tolist()[0]]
        #     epochs += [epoch]
        # data = {'value': values, 'epoch': epochs}
        # df = pd.DataFrame(data)
        # print(df)
        # plot = sns.FacetGrid(df, row=None, col=None, sharey=False)
        # plot.map_dataframe(sns.lineplot, x='epoch', y='value', data=data, palette=palette)
        # axes = plot.axes.flatten()
        # axes_title = ['','']
        # axes_y_labels = ['Accuracy']
        # axes_x_labels = ['Epoch']
        # for idx, ax in enumerate(axes):
        #     ax.set_title(axes_title[idx])
        #     ax.set_ylabel(axes_y_labels[idx])
        #     ax.set_xlabel(axes_x_labels[idx])
        #     # ax.lines[0].set_color('black')
        #     # ax.lines[0].set_linestyle('-')
        #     # ax.lines[1].set_color('red')
        #     # ax.lines[1].set_linestyle('-')
        #     # add vertical line to define best checkpoint
        #     ax.axvline(x=best_epoch, color='blue', linestyle='--', linewidth=2)
        #     ax.set_ylim(0,1)
        # plot.add_legend(loc='lower center')
        # plt.savefig(os.path.join(args.testing_dir, 'testing/dataset', 'accuracy.png'), dpi=300)



    # # create plots for one model tested across multiple checkpoints
    # if args.testing_dir is not None:
    #     # get input data
    #     metrics_file = sorted(glob.glob(os.path.join(args.testing_dir, 'testing/dataset/*/metrics.tsv')))
    #     summary_file = sorted(glob.glob(os.path.join(args.testing_dir, 'testing/dataset/*/summary.tsv')))
    #     assert len(metrics_file) == len(summary_file)
    #     best_epoch = glob.glob(os.path.join(args.testing_dir, 'model/*-best'))[0]
    #     best_epoch = int(best_epoch.split('-')[-2]) + 1
    #     print('best epoch', best_epoch)
    #     # create plots for precision and recall
    #     values = []
    #     metrics = []
    #     labels = []
    #     epochs = []
    #     for i in range(len(metrics_file)):
    #         print(i, metrics_file[i])
    #         print(metrics_file[i].split('/')[-2].split('-'))
    #         if len(metrics_file[i].split('/')[-2].split('-')) > 1:
    #             epoch = int(metrics_file[i].split('/')[-2].split('-')[0])
    #         else:
    #             epoch = int(metrics_file[i].split('/')[-2])
    #         print(epoch)
    #         metrics_df =  pd.read_csv(metrics_file[i], sep='\t', header=None)
    #         metrics_df.columns = ['label','metric','value']
    #         print(metrics_df)
    #         values += metrics_df['value'].tolist()
    #         metrics += metrics_df['metric'].tolist()
    #         labels += metrics_df['label'].tolist()
    #         epochs += metrics_df.shape[0]*[epoch]
        
    #     data = {'value': values, 'metric': metrics, 'label':labels, 'epoch': epochs}
    #     df = pd.DataFrame(data)
    #     df['label'].replace(0, 'label 0', inplace=True)
    #     df['label'].replace(1, 'label 1', inplace=True)
    #     print(df)
    #     palette = {'label 0': 'orange', 'label 1': 'pink'}
    #     plot = sns.FacetGrid(df, row=None, col='metric', sharey=False)
    #     plot.map_dataframe(sns.lineplot, x='epoch', y='value', data=data, hue='label', palette=palette)
    #     axes = plot.axes.flatten()
    #     axes_title = ['','']
    #     axes_y_labels = ['Precision', 'Recall']
    #     axes_x_labels = ['Epoch', 'Epoch']
    #     for idx, ax in enumerate(axes):
    #         ax.set_title(axes_title[idx])
    #         ax.set_ylabel(axes_y_labels[idx])
    #         ax.set_xlabel(axes_x_labels[idx])
    #         # ax.lines[0].set_color('black')
    #         # ax.lines[0].set_linestyle('-')
    #         # ax.lines[1].set_color('red')
    #         # ax.lines[1].set_linestyle('-')
    #         # add vertical line to define best checkpoint
    #         ax.axvline(x=best_epoch, color='blue', linestyle='--', linewidth=2)
    #         if idx == 0:
    #             ax.set_ylim(0,1)
    #         if idx == 1:
    #             ax.set_ylim(0,1)
    #     plot.add_legend(loc='lower center')
    #     plt.savefig(os.path.join(args.testing_dir, 'testing/dataset', 'metrics.png'), dpi=300)

    #     # plot accuracy
    #     values = []
    #     epochs = []
    #     for i in range(len(summary_file)):
    #         if len(metrics_file[i].split('/')[-2].split('-')) > 1:
    #             epoch = int(metrics_file[i].split('/')[-2].split('-')[0])
    #         else:
    #             epoch = int(metrics_file[i].split('/')[-2])
    #         summary_df =  pd.read_csv(summary_file[i], sep='\t', header=None)
    #         summary_df.columns = ['metric','value']
    #         values += [summary_df['value'].tolist()[0]]
    #         epochs += [epoch]
    #     data = {'value': values, 'epoch': epochs}
    #     df = pd.DataFrame(data)
    #     print(df)
    #     plot = sns.FacetGrid(df, row=None, col=None, sharey=False)
    #     plot.map_dataframe(sns.lineplot, x='epoch', y='value', data=data, palette=palette)
    #     axes = plot.axes.flatten()
    #     axes_title = ['','']
    #     axes_y_labels = ['Accuracy']
    #     axes_x_labels = ['Epoch']
    #     for idx, ax in enumerate(axes):
    #         ax.set_title(axes_title[idx])
    #         ax.set_ylabel(axes_y_labels[idx])
    #         ax.set_xlabel(axes_x_labels[idx])
    #         # ax.lines[0].set_color('black')
    #         # ax.lines[0].set_linestyle('-')
    #         # ax.lines[1].set_color('red')
    #         # ax.lines[1].set_linestyle('-')
    #         # add vertical line to define best checkpoint
    #         ax.axvline(x=best_epoch, color='blue', linestyle='--', linewidth=2)
    #         ax.set_ylim(0,1)
    #     plot.add_legend(loc='lower center')
    #     plt.savefig(os.path.join(args.testing_dir, 'testing/dataset', 'accuracy.png'), dpi=300)

    
    # if args.testing_sum_dir:
    #     # create learning curves
    #     # get input data
    #     metrics_files = sorted(glob.glob(os.path.join(args.testing_sum_dir, '*/*/testing/metrics.tsv')))
    #     summary_files = sorted(glob.glob(os.path.join(args.testing_sum_dir, '*/*/testing/summary.tsv')))
    #     print(metrics_files)
    #     print(summary_files)
    #     assert len(metrics_files) == len(summary_files)
    #     batch_size = []
    #     values = []
    #     for i in range(len(summary_files)):
    #         bs = summary_files[i].split('/')[-3].split('-')[-1]
    #         acc_df = pd.read_csv(summary_files[i], sep='\t', header=None)
    #         print(bs)
    #         print(acc_df.iloc[0, 1])
    #         values.append(acc_df.iloc[0, 1])
    #         batch_size.append(int(bs))

    #     data = {'accuracy': values, 'batch_size': batch_size}
    #     df = pd.DataFrame(data)
    #     plt.figure(figsize=(5, 5))
    #     sns.set_color_codes('pastel')
    #     plot = sns.barplot(df, x='batch_size', y='accuracy', legend=False, color='b', width=0.7)
    #     plot.set_ylabel('Accuracy')
    #     plot.set_xlabel('Batch size')
    #     plot.set_ylim(0,1)
    #     plt.savefig(os.path.join(args.testing_sum_dir, 'accuracy.png'), dpi=300)

    #     batch_size = []
    #     labels = []
    #     values = []
    #     metrics = []
    #     for i in range(len(metrics_files)):
    #         bs = summary_files[i].split('/')[-3].split('-')[-1]
    #         metrics_df = pd.read_csv(metrics_files[i], sep='\t', header=None)
    #         print(metrics_df.iloc[0, 2])
    #         label_1_prec = metrics_df.iloc[0, 2]
    #         label_0_prec = metrics_df.iloc[1, 2]
    #         label_1_rec = metrics_df.iloc[2, 2]
    #         label_0_rec = metrics_df.iloc[3, 2]
    #         values += [label_1_prec, label_0_prec, label_1_rec, label_0_rec]
    #         metrics += ['precision', 'precision', 'recall', 'recall']
    #         batch_size += [int(bs)]*4
    #         labels += [1, 0, 1, 0]

    #     data = {'values': values, 'batch_size': batch_size, 'metrics': metrics, 'labels': labels}
    #     df = pd.DataFrame(data)
    #     print(df)
    #     plot = sns.FacetGrid(df, row='metrics', col='labels', sharey=False)
    #     plot.map_dataframe(sns.barplot, x='batch_size', y='values', color='b')
    #     axes = plot.axes.flatten()
    #     axes_title = ['Positive class','Negative class', '', '']
    #     axes_y_labels = ['Precision', '', 'Recall', '']
    #     axes_x_labels = ['', '', 'Batch size', 'Batch size']
    #     for idx, ax in enumerate(axes):
    #         ax.set_title(axes_title[idx])
    #         ax.set_ylabel(axes_y_labels[idx])
    #         ax.set_xlabel(axes_x_labels[idx])
    #         ax.set_ylim(0,1)
    #         print(idx, ax.get_title(), ax.get_ylabel(), ax.get_xlabel(), ax.get_ylim())
    #     plt.savefig(os.path.join(args.testing_sum_dir, 'metrics.png'), dpi=300)


        #     train_df = pd.read_csv(training_files[i], sep='\t', header=None)
        #     val_df = pd.read_csv(validation_files[i], sep='\t', header=None)
        #     num_epochs, _ = val_df.shape
        #     num_steps_per_epoch = train_df.iloc[:, 1].tolist()[-1]
        #     train_accuracy = [train_df.iloc[:, 3].tolist()[j] for j in range(0, len(train_df), num_steps_per_epoch)]
        #     train_loss = [train_df.iloc[:, 2].tolist()[j] for j in range(0, len(train_df), num_steps_per_epoch)]
        #     values += train_accuracy
        #     values += train_loss
        #     dataset += ['training']*(len(train_accuracy)*2)
        #     metric += ['accuracy']*len(train_accuracy) + ['loss']*len(train_loss)
        #     val_accuracy = val_df.iloc[:, 3].tolist()
        #     val_loss = val_df.iloc[:, 2].tolist()
        #     values += val_accuracy
        #     values += val_loss
        #     batch_size += [int(train_bs)]*(len(train_accuracy)*2+len(val_accuracy)*2)
        #     dataset += ['validation']*(len(val_accuracy)*2)
        #     metric += ['accuracy']*len(val_accuracy) + ['loss']*len(val_loss)
        #     epochs += list(range(1, num_epochs+1, 1))*4
        # # create dataframe
        # data = {'batch_size': batch_size, 'value': values, 'metric': metric, 'dataset': dataset, 'epoch': epochs}
        # df = pd.DataFrame(data)
        # max_loss = max(df.loc[df['metric'] == 'loss', 'value'].tolist())
        # min_loss = min(df.loc[df['metric'] == 'loss', 'value'].tolist())
        # print(df)
        # print(df.shape)
        # line_styles = ['-', '--']
        # palette = {'training': 'black', 'validation': 'red'}
        # plot = sns.FacetGrid(df, row='metric', col='batch_size', sharey=False)
        # plot.map_dataframe(sns.lineplot, x='epoch', y='value', data=data, hue='dataset', palette=palette)
        # axes = plot.axes.flatten()
        # axes_title = ['batch size: 32','batch size: 64', 'batch size: 128', 'batch size: 256', '', '', '', '']
        # axes_y_labels = ['Accuracy', '', '', '', 'Loss', '', '', '',]
        # axes_x_labels = ['', '', '', '', 'Epoch', 'Epoch', 'Epoch', 'Epoch']
        # for idx, ax in enumerate(axes):
        #     ax.set_title(axes_title[idx])
        #     ax.set_ylabel(axes_y_labels[idx])
        #     ax.set_xlabel(axes_x_labels[idx])
        #     ax.lines[0].set_color('black')
        #     ax.lines[0].set_linestyle('-')
        #     ax.lines[1].set_color('red')
        #     ax.lines[1].set_linestyle('-')
        #     if idx in [4,5,6,7]:
        #         ax.set_ylim(min_loss,max_loss)
        #     if idx in [0,1,2,3]:
        #         ax.set_ylim(0,100)
        #     print(idx, ax.get_title(), ax.get_ylabel(), ax.get_xlabel(), ax.get_ylim())
        # plot.add_legend()
        # # legend = plot.legend
        # # legend.set_loc('lower center')
        # # plt.tight_layout()
        # plt.savefig(os.path.join(args.lc_dir, 'learning_curves.png'), dpi=300)


