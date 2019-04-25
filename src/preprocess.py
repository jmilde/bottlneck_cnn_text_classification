from util_np import np, vpack, unison_shfl
import pandas as pd
import os
from util_sp import load_spm, spm
from util_io import save_txt, pform
import re
from hyperparameters import preprocess_param as params


def preprocess(path_data, name, train_val_ratio, seed, max_len):
    # load all parameters


    # PATH FOR SAVING DATA
    path_vocab = pform(path_data, "vocab")
    path_txt = pform(path_data, "spm_tweets.txt")
    path_trainlbl = pform(path_data, "train_lbl.npy")
    path_traintxt = pform(path_data, "train_txt.txt")

    # create new folder to save preprocessed data
    if not os.path.exists(path_data):
        os.makedirs(path_data)



    ### process data
    path = pform(path_data, 'labeled_data.csv')
    data = pd.read_csv(path, encoding='utf-8')
    txt, labels = [], []
    rx = "\n"
    for idx, tweet in enumerate(data['tweet'].values):
        label = data['class'].values[idx]  # 0,1,2 (hatespeech,offensive,neither)
        if label == 0 or label == 1:
            txt.append(re.sub(rx, "", tweet))
            labels.append(1)
        if label == 2:
            txt.append(re.sub(rx, "", tweet))
            labels.append(0)


    # shuffle all tweets and labels
    txt, labels = unison_shfl(txt, labels, seed=seed)

    # train sentence piece model once on all data
    save_txt(path_txt, txt)
    spm(name=path_vocab, path=path_txt)

    #  if we evalutate on a subset of all data
    train_size = int(len(txt)*train_val_ratio)  #train/test split
    valid_lbl = labels[train_size:]
    valid_txt = txt[train_size:]
    txt = txt[:train_size]
    labels = labels[:train_size]


    # save train data
    save_txt(path_traintxt, txt)
    np.save(path_trainlbl, labels)


    # load the trained sentence piece model and encode the validation sentences
    vocab = load_spm(path_vocab + ".model")
    ids, txt, lbl = [], [], []
    for idx, t in enumerate(valid_txt):
        s_id = vocab.encode_as_ids(t)
        if len(s_id) <= max_len: # only tweets with less than 250 splits
            ids.append(s_id)
            txt.append(tweet)
            lbl.append(valid_lbl[idx])

    # save valid data
    valid_txt = vpack(ids, (len(ids), max(map(len, ids))), 1, np.int32)
    np.save(path_data + '/valid_txt.npy', valid_txt)
    save_txt(path_data + '/valid_txt.txt', txt)
    np.save(path_data + '/valid_lbl.npy', lbl)


if __name__ == "__main__":
    for k, v in params.items():
        exec(k + "=v")
    preprocess(path_data, name, train_val_ratio, seed, max_len)
