import glob
import json
import matplotlib.pyplot as plt
import statistics


def get_values(json_filename):
    with open(json_filename, 'r') as f:
        json_dict = json.load(f)
    return json_dict


def update_dict(global_dict, local_dict):
    for k, v in local_dict.items():
        if k in global_dict:
            global_dict[k] += v
        else:
            global_dict[k] = v
    return global_dict


if __name__ == "__main__":
    json_files = sorted(glob.glob("*.json"))
    print(json_files)


    num_epochs = len(set([i.split('-')[1] for i in json_files]))
    print(num_epochs)
    num_gpus = len(set([i.split('-')[0] for i in json_files]))
    print(num_epochs, num_gpus)

    bins = 20

    fig1, ax1 = plt.subplots(nrows=1, ncols=1)

    for i in range(1, num_epochs+1, 1):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        epoch_sp_count = {}
        for j in range(num_gpus):
            gpu_sp_count = get_values(f'{j}-{i}-labels.json')
            print(f'epoch: {i} - gpu: {j} - {statistics.mean(gpu_sp_count.values())}')
            epoch_sp_count = update_dict(epoch_sp_count, gpu_sp_count)
            ax.hist(gpu_sp_count.values(), bins, alpha=0.5, label=f'gpu {j}')
        ax.set_xlabel('Number of reads per species')
        ax.set_ylabel('Count')
        ax.legend(loc='upper right')
        fig.savefig(f'species-hist-epoch-{i}.png')
        ax1.hist(epoch_sp_count.values(), bins, alpha=0.5, label=f'epoch {i}')
    ax1.set_xlabel('Number of reads per species')
    ax1.set_ylabel('Count')
    ax1.legend(loc='upper right')
    fig1.savefig('species-hist-all-gpus.png')





