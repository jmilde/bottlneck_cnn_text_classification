from util_io import pform
from model import classifier
from tqdm import trange
import util_sp as sp
from util_np import np, partition
from util_tf import tf, pipe, batch
import os
from hyperparameters import params

def train(batch_size, batch_valid, dim_emb, btlnk_dim, cnn_dim, att_dim, layers, keep_prob,
          gpu, path_data, path_ckpt, path_log, seed, trial, pretrain, max_len):

    # pick gpu
    if gpu != None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # form all path
    path_vocab = pform(path_data, "vocab.model")
    path_traintxt = pform(path_data, "train_txt.txt")
    path_trainlbl = pform(path_data, "train_lbl.npy")
    path_validtxt = pform(path_data, "valid_txt.npy")
    path_validlbl = pform(path_data, "valid_lbl.npy")

    ### LOAD
    # load validation data
    valid_txt = np.load(path_validtxt)
    valid_lbl = np.load(path_validlbl)
    # load sentence piece model
    vocab = sp.load_spm(path_vocab)
    dim_voc = vocab.GetPieceSize()

    # build pipeline
    batch_fn = lambda: batch(batch_size, path_traintxt, path_trainlbl, vocab, 25, 1, max_len)
    data = pipe(batch_fn, (tf.int32, tf.int32), prefetch=4)

    # create model
    model = classifier(data, dim_voc, dim_emb, cnn_dim, btlnk_dim, att_dim, layers, keep_prob, batch_size, True)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()

    if pretrain == None:
        # train new model
        tf.global_variables_initializer().run()
    else:
        # or restore pretrained model
        saver.restore(sess, pform(path_ckpt, pretrain))

    ### TENSORBOARRD
    summary_train = tf.summary.merge((tf.summary.scalar('train_acc', model['acc']),
                                      tf.summary.scalar('train_loss', model['loss'])))
    summary_test = tf.summary.merge((tf.summary.scalar('test_acc', model['acc']),
                                     tf.summary.scalar('test_loss', model['loss'])))
    wrtr = tf.summary.FileWriter(pform(path_log, trial))
    wrtr.add_graph(sess.graph)

    def summ(step):
        fetches = model['acc'], model['loss']
        results = map(np.mean, zip(*(
            sess.run(fetches, {model['tweets']: valid_txt[i:j], model['labels']: valid_lbl[i:j], model['train']:False})
            for i, j in partition(len(valid_txt), batch_valid, discard=False))))
        results = dict(zip(fetches, results))
        wrtr.add_summary(sess.run(summary_test, results), step)
        wrtr.add_summary(sess.run(summary_train), step)
        wrtr.flush()
        return list(results.values())

    ### TRAINING
    for idx in range(10000):
        for _ in trange(100, ncols=70):
            sess.run(model['train_step'])
        step = sess.run(model['step'])
        results = summ(step)
        print("{}: valid_acc: {:.3f}, valid_loss: {:.3f}".format(idx, results[0], results[1]))
        saver.save(sess, pform(path_ckpt, trial), write_meta_graph=False)


if __name__ == '__main__':
    for key, val in params.items():
        exec(key+"=val")

    train(batch_size, batch_valid, dim_emb, btlnk_dim, cnn_dim, att_dim, layers, keep_prob,
          gpu, path_data, path_ckpt, path_log, seed, trial, pretrain, max_len)
