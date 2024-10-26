import argparse
import random
import numpy as np
import os
from os.path import join
import const_rf as const
import multiprocessing as mp
import pandas as pd
from importlib import import_module
import tqdm
import sys
from os.path import join, dirname, abspath, pardir

# Add the RF directory to PYTHONPATH
BASE_DIR = abspath(join(dirname(__file__), pardir))
sys.path.append(BASE_DIR)

def parallel(para_list, n_jobs=1):
    pool = mp.Pool(n_jobs)
    data_dict = tqdm.tqdm(pool.imap(extract_feature, para_list), total=len(para_list))
    pool.close()
    return data_dict


def extract_feature(para):
    f, feature_func = para
    file_name = f.split('/')[-1]

    with open(f, 'r') as f:
        tcp_dump = f.readlines()

    seq = pd.Series(tcp_dump[:const.max_trace_length]).str.slice(0, -1).str.split(const.split_mark, expand=True).astype("float")
    times = np.array(seq.iloc[:, 0])
    length_seq = np.array(seq.iloc[:, 1]).astype("int")
    fun = import_module('FeatureExtraction.' + feature_func)
    feature = fun.fun(times, length_seq)
    if '-' in file_name:
        label = file_name.split('-')[0]
        label = int(label)
    else:
        label = const.MONITORED_SITE_NUM

    return feature, label


def process_dataset(traces_path):
    output_dir = const.output_dir + defence + '-' + feature_func

    para_list = []

    for i in range(const.MONITORED_SITE_NUM):
        for j in range(const.MONITORED_INST_NUM):
            file_path = f"{traces_path}{i}-{j}.cell"
            if os.path.exists(file_path):
                para_list.append((file_path, feature_func))

    if const.OPEN_WORLD:
        for i in range(const.UNMONITORED_SITE_NUM):
            file_path = join(traces_path, f"{i}.cell")
            if os.path.exists(file_path):
                para_list.append((file_path, feature_func))

    random.shuffle(para_list)

    data_dict = {'dataset': [], 'label': []}
    raw_data_dict = parallel(para_list, n_jobs=15)

    features, label = zip(*raw_data_dict)

    features = np.array(features)
    if len(features.shape) < 3:
        features = features[:, np.newaxis, :]
    labels = np.array(label)

    print("dataset shape:{}, label shape:{}".format(features.shape, labels.shape))
    data_dict['dataset'], data_dict['label'] = features, labels
    np.save(output_dir, data_dict)

    print(f'Saved to {output_dir}.npy')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process dataset for training.')
    parser.add_argument('--traces_path', type=str, required=True, help='Path to the directory containing trace files.')
    args = parser.parse_args()

    defence = 'Undefended'
    feature_func = 'packets_per_slot'

    process_dataset(args.traces_path)
