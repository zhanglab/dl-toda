# DL-TODA: a deep learning tool for omics data analysis

DL-TODA classifies metagenomic data into 3313 bacterial species with a convolutional neural network. The present GitHub page contains all scripts used for simulating reads, preparing the datasets, training and testing DL-TODA and classify metagenomes.


**_Citation: Cres C, Tritt A, Bouchard K, Zhang Y. DL-TODA: A Deep Learning Tool for Omics Data Analysis. *Under Review.*_**


## Installation of DL-TODA
DL-TODA has been designed to run with NVIDIA GPUs.

1. Clone dl-toda repository

` git clone https://github.com/zhanglab/dl-toda.git`

2. Create a conda environment to run DL-TODA

The installation of DL-TODA uses poetry to manage dependencies. The step to install poetry is necessary to install dali-tf plugin and tensorflow properly.

`conda env create --name dltoda --file=environment.yml`

`conda activate dltoda`

`export HOROVOD_WITH_TENSORFLOW=1`

`poetry install`

`poetry install`

Set up library paths:

`mkdir -p $CONDA_PREFIX/etc/conda/activate.d`

`echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' >
$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh`


## Pipeline

![alt text](https://i.imgur.com/eTzeqJW.jpg)



## Data preparation

1. Simulate reads

Paired-end 250 bp reads can be simulated for a genome using the python script run_art_sim.py provided that ART Illumina (add link) has been previously installed.  

Use the following command to simulate reads for all training genomes:
`cat training_genomes.tsv | parallel -j 20 python /dl-toda/dataprep_scripts/run_art_sim.py {} <coverage> <output directory> <path to directory containing fasta file> <path to art illumina executable>`

The input data is a line consisting of the genome accession id separated by a tab from an integer corresponding to the species it belongs to (example: "GCA_000010565.1	0"). See training_genomes.tsv and testing_genomes.tsv available here (add figshare link).

3. Training dataset

Reads simulated for training have to be shuffled and randomly assigned for training or validation. The script datasets.py will process each fastq file from a genome separately and split forward and reverse reads for training (70%) and validation (30%). The output consists of fastq files for training or validation of 20,000,000 shuffled reads.

The following command will generate the training and validation fastq files:
`python /dl-toda/dataprep_scripts/datasets.py <input directory containing fastq files> <output directory> training_genomes.tsv species_labels.json`

3. Testing dataset

Reads simulated for testing are directly converted into Tensorflow TFRecords.

4. Metagenomic data

Reads obtained from DNA sequencing are directly converted into Tensorflow TFRecords.

4. Convert reads to TFRecords

Reads stored in fastq files for training, validation or testing are converted to TFRecords using the following command:

`cat list_fastq_files.txt | parallel -j 20 /dl-toda/DL_scripts/create_tfrecords.py --input_fastq {} --vocab 12mers.txt --output_dir <output directory> --dataset_type 'sim'`

The `dataset_type` parameter has to be set to `'meta'` in order to convert fastq files with metagenomic reads to TFRecords.

Notes: 
* list_fastq_files.txt is a text file with the full path of a fastq file per line .
* 12mers.txt can be found here (add figshare link).

5. Create NVIDIA DALI index files

An index has to be created for each TFRecord file using the script tfrecord2idx provided by the NVIDIA Data Loading Library. An example for running the script is given below.

`cat list_tfrec_files.txt | parallel -j 20 tfrecord2idx {} {}.idx`

Note: 
* list_tfrec_files.txt is a text file with the full path of a TFRecord file per line.

6. Recommendations

GNU parallel (https://www.gnu.org/software/parallel/) is recommended to expedite the simulation of reads and the generation of TFRecord files. The number of input file to process in parallel is set to 20 here but can be adjusted depending on the number of CPU available.

## Training

`
mpirun -np <number of GPUs> -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python read_classifier_training.py --tfrecords <path to directory containing TFRecord files> --dali_idx <path to directory containing NVIDIA DALI index files files> --data_type 'sim'
`

## Testing

`
mpirun -np <number of GPUs> -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python read_classifier.py --tfrecords <path to directory containing TFRecord files> --dali_idx <path to directory containing NVIDIA DALI index files files> --data_type 'sim' --ckpt <path to checkpoint file of DL-TODA model saved at epoch 14>
`

## Read classification

`
mpirun -np <number of GPUs> -bind-to none -map-by slot -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -mca pml ob1 -mca btl ^openib python read_classifier.py --tfrecords <path to directory containing TFRecord files> --dali_idx <path to directory containing NVIDIA DALI index files files> --data_type 'meta' --ckpt <path to checkpoint file of DL-TODA model saved at epoch 14>
`

Notes:
* The batch size is 8192 (for testing and classifying metagenomes) or 512 (for training) by default and can be adjusted using `--batch_size`. 
* The checkpoint of DL-TODA model saved at epoch 14 can be found here (add figshare link). It consists of two files (ckpts-14.data-00000-of-00001 and ckpts-14.index) that should be uploaded and stored in the same directory.


