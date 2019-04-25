params = {
    "trial": "testtrial",  # name your trial
    "gpu": None,  # pick gpu, if available (terminal: nivida-smi)
    "pretrain": None,  # if loading a pretrained model, give name of the checkpoint you want to laod
    "path_data": "../data/",
    "path_ckpt": "../ckpt/",
    "path_log": "../log/",
    "batch_size": 64,  # training batch_size
    "batch_valid": 128,  # batchsize for evalaution
    "max_len": 250, # max length of the text sequence (# of sentence piece splits)
    "dim_emb": 256,  # embedding dimensions
    "btlnk_dim": 64,  # size of the bottleneck cnn
    "cnn_dim": 256,  # cnn dimensionality
    "att_dim": 128,  # attention dimensionality
    "layers": 12,  # number of cnn bottleneck layers
    "keep_prob": 0.5,  # probabilty of keeping information when using dropout
    "seed": 25}


preprocess_param = {
    "path_data": "../data/",  # path to main data folder
    "name": "hatespeech",  # name under which the preprocessed data is saved
    "train_val_ratio": 0.8,  # if valid_on_og = False, pick how to split of the testset from all data
    "seed": 25,
    "max_len": 250
}
