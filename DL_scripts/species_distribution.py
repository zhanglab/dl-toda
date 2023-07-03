import glob
import json
import matplotlib.pyplot as plt


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


    num_epochs = list(set([i.split('-')[1] for i in json_files]))
    num_gpus = list(set([i.split('-')[0] for i in json_files]))
    print(num_epochs, num_gpus)


    for i in range(num_epochs):
        epoch_sp_count = {}
        for j in range(num_gpus):
            gpu_sp_count = get_values(f'{j}-{i}-labels.json')
            epoch_sp_count = update_dict(epoch_sp_count, gpu_sp_count)
        plt.hist(epoch_sp_count.values(), label=f'epoch {i}')
    plt.legend(loc='upper right')
    plt.savefig('species-hist.png')


