import datetime
import tensorflow as tf
import horovod.tensorflow as hvd
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.tfrecord as tfrec
import nvidia.dali.plugin.tf as dali_tf
from AlexNet import AlexNet
from lstm import LSTM
from VDCNN import VDCNN
from VGG16 import VGG16
from DNA_model_1 import DNA_net_1
from DNA_model_2 import DNA_net_2
from transformers import TFBertForSequenceClassification, BertConfig
import os
import sys
import json
import glob
import time
import numpy as np
import math
import argparse
import random


# set seed
seed = 42
# set seed for tensorflow
tf.random.set_seed(seed)
# set seed for numpy operations
np.random.seed(seed)
# set the global python random seed
random.seed(seed)
# activate tensorflow deterministic behavior
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# set the number of threads used for parallel execution of independent operations to 1
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

dl_toda_dir = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[0:-1])

# disable eager execution
# tf.compat.v1.disable_eager_execution()
print(f'Is eager execution enabled: {tf.executing_eagerly()}')

# print which unit (CPU/GPU) is used for an operation
#tf.debugging.set_log_device_placement(True)

# enable XLA = XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear algebra that can accelerate
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

# define the DALI pipeline for CNN
@pipeline_def
def dali_pipeline(tfrec_filenames, tfrec_idx_filenames, shard_id, initial_fill, num_gpus, training=True):
    stick_to_shard = True
    inputs = fn.readers.tfrecord(path=tfrec_filenames,
                                 index_path=tfrec_idx_filenames,
                                 random_shuffle=training,
                                 shard_id=shard_id,
                                 num_shards=num_gpus,
                                 initial_fill=initial_fill,
                                 stick_to_shard=stick_to_shard,
                                 features={
                                     "read": tfrec.VarLenFeature([], tfrec.int64, 0)})
    # retrieve reads and labels and copy them to the gpus
    reads = inputs["read"].gpu()
    return reads

# define the DALI pipeline for BERT 
@pipeline_def
def bert_dali_pipeline(tfrec_filenames, tfrec_idx_filenames, shard_id, initial_fill, num_gpus, training=True):
    stick_to_shard = True
    inputs = fn.readers.tfrecord(path=tfrec_filenames,
                                 index_path=tfrec_idx_filenames,
                                 random_shuffle=training,
                                 shard_id=shard_id,
                                 num_shards=num_gpus,
                                 stick_to_shard=stick_to_shard,
                                 initial_fill=initial_fill,
                                 features={
                                     "input_ids": tfrec.VarLenFeature([], tfrec.int64, 0),
                                     "attention_mask": tfrec.VarLenFeature([], tfrec.int64, 0),
                                     "position_ids": tfrec.VarLenFeature([], tfrec.int64, 0),
                                     "token_type_ids": tfrec.VarLenFeature([], tfrec.int64, 0)})
    
    # retrieve data and copy it to the gpus
    input_ids = inputs["input_ids"].gpu()
    attention_mask = inputs["attention_mask"].gpu()
    token_type_ids = inputs["token_type_ids"].gpu()
    position_ids = inputs["position_ids"].gpu()

    return (input_ids, attention_mask, position_ids, token_type_ids)


class DALIPreprocessor(object):
    def __init__(self, model_type, filenames, idx_filenames, batch_size, vector_size, initial_fill, deterministic=False, training=False):

        device_id = hvd.local_rank()
        shard_id = hvd.rank()
        num_gpus = hvd.size()
        
        # self.batch_size = batch_size
        # self.device_id = device_id

        if model_type == "BERT":
            self.pipe = bert_dali_pipeline(tfrec_filenames=filenames, tfrec_idx_filenames=idx_filenames, batch_size=batch_size,
                                      device_id=device_id, shard_id=shard_id, initial_fill=initial_fill, num_gpus=num_gpus,
                                      training=training, seed=7 * (1 + hvd.rank()) if deterministic else None)

            self.dalidataset = dali_tf.DALIDataset(fail_on_device_mismatch=False, pipeline=self.pipe,
                output_shapes=((batch_size, vector_size), (batch_size, vector_size), (batch_size, vector_size), (batch_size, vector_size)),
                batch_size=batch_size, output_dtypes=(tf.int64, tf.int64, tf.int64, tf.int64), device_id=device_id)
        else:
            self.pipe = dali_pipeline(tfrec_filenames=filenames, tfrec_idx_filenames=idx_filenames, batch_size=batch_size,
                                      device_id=device_id, shard_id=shard_id, initial_fill=initial_fill, num_gpus=num_gpus,
                                      training=training, seed=7 * (1 + hvd.rank()) if deterministic else None)
   
            self.dalidataset = dali_tf.DALIDataset(fail_on_device_mismatch=False, pipeline=self.pipe,
                output_shapes=((batch_size, vector_size)),
                batch_size=batch_size, output_dtypes=(tf.int64), device_id=device_id)

    def get_device_dataset(self):
        return self.dalidataset


def build_dataset(args, filenames, num_classes, is_training, drop_remainder):

    def load_tfrecords_for_dltoda(proto_example):
        data_description = {
            'read': tf.io.VarLenFeature(tf.int64)
        }
        # load one example
        parsed_example = tf.io.parse_single_example(serialized=proto_example, features=data_description)
        read = parsed_example['read']
        read = tf.sparse.to_dense(read)
        return read

    def load_tfrecords_for_bert(proto_example):
        name_to_features = {
          "input_ids": tf.io.FixedLenFeature([args.vector_size], tf.int64),
          "attention_mask": tf.io.FixedLenFeature([args.vector_size], tf.int64),
          "position_ids": tf.io.FixedLenFeature([args.vector_size], tf.int64),
          "token_type_ids": tf.io.FixedLenFeature([args.vector_size], tf.int64)
        }
        parsed_example = tf.io.parse_single_example(serialized=proto_example, features=name_to_features)

        return {"input_ids": parsed_example['input_ids'], "position_ids": parsed_example['position_ids'], "token_type_ids": parsed_example['token_type_ids'], "attention_mask": parsed_example['attention_mask']}

    """ Return data in TFRecords """
    fn_load_data = {'DLTODA': load_tfrecords_for_dltoda, 'BERT': load_tfrecords_for_bert}

    dataset = tf.data.TFRecordDataset([filenames])

    if is_training:
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.map(map_func=fn_load_data[args.model_type])
    dataset = dataset.batch(args.batch_size, drop_remainder=drop_remainder)


    # Load data as shards
    # dataset = tf.data.Dataset.list_files(tfrecord_path)
    # dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x), num_parallel_calls=tf.data.experimental.AUTOTUNE,
                                 # deterministic=False)
    # dataset = dataset.map(map_func=fn_load_data[datatype], num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # dataset = dataset.padded_batch(batch_size,
                                   # padded_shapes=(tf.TensorShape([vector_size]), tf.TensorShape([num_classes])),)
    # dataset = dataset.cache()
    # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

@tf.function
def testing_step(model_type, data, model, nvidia_dali=False):
    training = False

    if model_type == 'BERT':
        if nvidia_dali:
            input_ids, attention_mask, position_ids, token_type_ids = data
        else:
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            token_type_ids = data["token_type_ids"]
            position_ids = data["position_ids"]

        # outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        outputs = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # outputs = model(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, labels=labels)
        # outputs = model(**data)
        # logits = model(**data).logits
        # loss_value = model(**data).loss
        # predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        # probs = tf.nn.softmax(logits, axis=-1)
        # labels = data["labels"]
        logits = outputs.logits
        probs = tf.nn.softmax(logits, axis=-1)
    else:
        reads = data
        probs = model(reads, training=training)
    
    # get predicted labels and confidence scores
    pred_labels = tf.math.argmax(probs, axis=1)
    if tf.shape(probs)[1] == 2:
        pred_probs = probs
    else:
        pred_probs = tf.reduce_max(probs, axis=1)

    return pred_labels, pred_probs


def main():
    start = datetime.datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument('--tfrecords', type=str, help='path to tfrecords', required=True)
    parser.add_argument('--output_dir', type=str, help='directory to store results', default=os.getcwd())
    parser.add_argument('--init_lr', type=float, help='initial learning rate', default=0.0001)
    parser.add_argument('--batch_size', type=int, help='batch size per gpu', default=8192)
    parser.add_argument('--DNA_model', action='store_true', default=False)
    parser.add_argument('--n_rows', type=int, default=50)
    parser.add_argument('--n_cols', type=int, default=5)
    parser.add_argument('--nvidia_dali', action='store_true', default=False)
    parser.add_argument('--k_value', type=int, help='length of kmer strings', default=12)
    parser.add_argument('--embedding_size', type=int, help='size of embedding vectors', default=60)
    parser.add_argument('--dropout_rate', type=float, help='dropout rate to apply to layers', default=0.7)
    parser.add_argument('--vector_size', type=int, help='size of input vectors')
    parser.add_argument('--vocab', help="Path to the vocabulary file", required=('AlexNet' in sys.argv))
    parser.add_argument('--model_type', type=str, help='type of model', choices=['DNA_1', 'DNA_2', 'AlexNet', 'VGG16', 'VDCNN', 'LSTM', 'BERT'])
    parser.add_argument('--bert_config_file', type=str, help='path to bert config file', required=('BERT' in sys.argv))
    parser.add_argument('--model', type=str, help='path to directory containing model in SavedModel format')
    parser.add_argument('--class_mapping', type=str, help='path to json file containing dictionary mapping taxa to labels', default=os.path.join(dl_toda_dir, 'data', 'species_labels.json'))
    parser.add_argument('--ckpt', type=str, help='path to checkpoint file (only add the prefix)')
    parser.add_argument('--pretrained', type=str, help='path to directory containing hf pretrained model saved using save_pretrained')
    parser.add_argument('--max_read_size', type=int, help='maximum read size in training dataset', default=250)
    parser.add_argument('--initial_fill', type=int, help='size of the buffer for random shuffling', default=10000)
    args = parser.parse_args()

    # Initialize Horovod
    hvd.init()
    # Map one GPU per process
    # use hvd.local_rank() for gpu pinning instead of hvd.rank()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f'GPU RANK: {hvd.rank()}/{hvd.local_rank()} - LIST GPUs: {gpus}')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        # tf.config.experimental.set_visible_devices(gpus, 'GPU')
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    models = {'DNA_1': DNA_net_1, 'DNA_2': DNA_net_2, 'AlexNet': AlexNet, 'VGG16': VGG16, 'VDCNN': VDCNN, 'LSTM': LSTM}

    # get vocabulary size
    if args.model_type != 'BERT':
        with open(f'{args.vocab}/{args.k_value}mers.txt', 'r') as f:
            content = f.readlines()
            vocab_size = len(content)

    # load class_mapping file mapping label IDs to species
    if args.class_mapping:
        f = open(args.class_mapping)
        class_mapping = json.load(f)
        num_labels = len(class_mapping)
    else:
        num_labels = args.num_labels

    # create dtype policy
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print('Compute dtype: %s' % policy.compute_dtype)
    print('Variable dtype: %s' % policy.variable_dtype)
    
    if hvd.rank() == 0:
        # create output directories
        if not os.path.isdir(args.output_dir):
            os.makedirs(os.path.join(args.output_dir))

    if args.model_type == 'BERT':
        with open(args.bert_config_file, "r") as f:
                args.config_dict = json.load(f)
        args.vector_size = args.config_dict['max_position_embeddings']
        if args.model is not None:
            model = tf.keras.models.load_model(args.model)
        else:
            bert_config = BertConfig(vocab_size=args.config_dict["vocab_size"])
            model = TFBertForSequenceClassification.from_pretrained(args.pretrained, config=bert_config)
    else:
        if args.model is not None:
            model = tf.keras.models.load_model(args.model)
        elif args.ckpt is not None:
            model = models[args.model_type](args, args.vector_size, args.embedding_size, num_labels, vocab_size, args.dropout_rate)
            # define the optimizer
            opt = tf.keras.optimizers.Adam(args.init_lr)
            # prevent numeric underflow when using float16
            opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
            checkpoint = tf.train.Checkpoint(optimizer=opt, model=model)
            checkpoint.restore(args.ckpt).expect_partial()

    # get list of testing tfrecords, number of reads per tfrecords and reads id for metagenomic data
    test_files = sorted(glob.glob(os.path.join(args.tfrecords, '*.tfrec')))
    num_reads_files = sorted(glob.glob(os.path.join(args.tfrecords, '*-read_count')))
    read_ids_files = sorted(glob.glob(os.path.join(args.tfrecords, '*-read_ids.tsv')))

    if args.nvidia_dali:
        # get nvidia dali indexes
        test_idx_files = sorted(glob.glob(os.path.join(args.tfrecords, 'idx_files', '*.idx')))
    
    # split tfrecords between gpus
    test_files_per_gpu = len(test_files)//hvd.size()

    if hvd.rank() != hvd.size() - 1:
        gpu_test_files = test_files[hvd.rank()*test_files_per_gpu:(hvd.rank()+1)*test_files_per_gpu]
        gpu_num_reads_files = num_reads_files[hvd.rank()*test_files_per_gpu:(hvd.rank()+1)*test_files_per_gpu]
        gpu_read_ids_files = read_ids_files[hvd.rank()*test_files_per_gpu:(hvd.rank()+1)*test_files_per_gpu] if len(read_ids_files) != 0 else None

        if args.nvidia_dali:
            gpu_test_idx_files = test_idx_files[hvd.rank()*test_files_per_gpu:(hvd.rank()+1)*test_files_per_gpu]
    else:
        gpu_test_files = test_files[hvd.rank()*test_files_per_gpu:len(test_files)]
        gpu_num_reads_files = num_reads_files[hvd.rank()*test_files_per_gpu:len(test_files)]
        gpu_read_ids_files = read_ids_files[hvd.rank()*test_files_per_gpu:len(test_files)] if len(read_ids_files) != 0 else None

        if args.nvidia_dali:
            gpu_test_idx_files = test_idx_files[hvd.rank()*test_files_per_gpu:len(test_files)]

    elapsed_time = []
    num_reads_classified = 0
    for i in range(len(test_files)):
        print(test_files[i])
        start_time = time.time()
        # get number of reads in test file
        with open(os.path.join(args.tfrecords, num_reads_files[i]), 'r') as f:
            num_reads = int(f.readline())
        print(f'number of reads to classify: {num_reads}')
        num_reads_classified += num_reads

        # compute number of steps required to iterate over entire test set
        test_steps = math.ceil(num_reads/(args.batch_size))

        # load data
        if args.nvidia_dali:
            nvidia_dali = True
            test_preprocessor = DALIPreprocessor(args.model_type, test_files[i], test_idx_files[i], args.batch_size, args.vector_size, args.initial_fill, deterministic=False, training=False)
            test_input = test_preprocessor.get_device_dataset()
        else:
            nvidia_dali=False
            test_input = build_dataset(args, test_files[i], num_labels, is_training=False, drop_remainder=False)

        # create empty arrays to store the predicted and true values, the confidence scores and the probability distributions
        # all_predictions = tf.zeros([args.batch_size, NUM_CLASSES], dtype=tf.dtypes.float32, name=None)
        all_pred_sp = [tf.zeros([args.batch_size], dtype=tf.dtypes.float32, name=None)]
        all_prob_sp = [tf.zeros([args.batch_size], dtype=tf.dtypes.float32, name=None)]
        # all_prob_labels = [tf.zeros([args.batch_size], dtype=tf.dtypes.float32, name=None)]
        
        for batch, data in enumerate(test_input.take(test_steps), 1):
            # batch_predictions, batch_pred_sp, batch_prob_sp = testing_step(args.data_type, reads, labels, model)
            batch_pred_sp, batch_prob_sp = testing_step(args.model_type, data, model)

            if batch == 1:
                all_pred_sp = [batch_pred_sp]
                all_prob_sp = [batch_prob_sp]
                # all_prob_labels = [batch_label_prob]
                # all_predictions = batch_predictions
            else:
                # all_predictions = tf.concat([all_predictions, batch_predictions], 0)
                all_pred_sp = tf.concat([all_pred_sp, [batch_pred_sp]], 1)
                all_prob_sp = tf.concat([all_prob_sp, [batch_prob_sp]], 1)
                # all_prob_labels = tf.concat([all_prob_labels, [batch_label_prob]], 1)

        # get list of true species, predicted species and predicted probabilities
        # all_predictions = all_predictions.numpy()
        all_pred_sp = all_pred_sp[0].numpy()
        all_prob_sp = all_prob_sp[0].numpy()
        # all_prob_labels = all_prob_labels[0].numpy()
        print(f'before adjusting: {len(all_pred_sp)}\t{len(all_prob_sp)}\n')


        # adjust the list of predicted species and read ids if necessary
        if len(all_labels) > num_reads:
            num_extra_reads = (test_steps*args.batch_size) - num_reads
            # all_predictions = all_predictions[:-num_extra_reads]
            all_pred_sp = all_pred_sp[:-num_extra_reads]
            all_prob_sp = all_prob_sp[:-num_extra_reads]
            print(f'number of reads: {num_extra_reads}\t{num_reads}\t{len(all_pred_sp)}\t{len(all_prob_sp)}\n')
            print(all_pred_sp[0], all_prob_sp[0])
            # all_prob_labels = all_prob_labels[:-num_extra_reads]

        # get dictionary mapping read ids to labels
        with open(os.path.join(args.tfrecords, read_ids_files[i]), 'r') as f:
            content = f.readlines()
            dict_read_ids = {content[j].rstrip().split('\t')[1]: '@' + content[j].rstrip().split('\t')[0] for j in range(len(content))}
        # write results to file
        with open(os.path.join(args.output_dir, f'{gpu_test_files[i].split("/")[-1].split(".")[0]}-out.tsv'), 'w') as out_f:
            for j in range(num_reads):
                out_f.write(f'{dict_read_ids[str(all_labels[j])]}\t\t{all_pred_sp[j]}\t{all_prob_sp[j]}\n')


        end_time = time.time()
        # elapsed_time = np.append(elapsed_time, end_time - start_time)
        elapsed_time.append(end_time - start_time)

    end = datetime.datetime.now()
    total_time = end - start
    hours, seconds = divmod(total_time.seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    with open(os.path.join(args.output_dir, f'testing-summary-{hvd.rank()}.tsv'), 'w') as outfile:
        outfile.write(f'{hvd.rank()}\t{args.batch_size}\t{hvd.size()}\t{hvd.rank()}\t{len(test_files)}\t{num_reads_classified}\t')
        if args.data_type == 'sim':
            outfile.write(f'{test_accuracy.result().numpy()}\t{test_loss.result().numpy()}\t')
        if args.ckpt:
            outfile.write(f'{args.ckpt}')
        # else:
        #     outfile.write(f'model saved at last epoch')
        outfile.write(f'\t{hours}:{minutes}:{seconds}:{total_time.microseconds}\t')

        if len(elapsed_time) > 1:
            outfile.write(f'{(num_reads_classified / sum(elapsed_time))} reads/sec\n')
        else:
            outfile.write(f'{(num_reads_classified / elapsed_time[0])} reads/sec\n')


if __name__ == "__main__":
    main()
