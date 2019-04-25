from util_io import load_txt, pform
from tqdm import trange
import util_sp as sp
from util_np import np, vpack, sample, partition
from util_tf import tf, pipe, placeholder, batch, conv1D, btlnk_residual, attention


def classifier(data, dim_voc, dim_emb, cnn_dim, btlnk_dim, att_dim, layers, keep_prob, batch_size, train, eos=1):
    # dim_voc   : vocab size
    # dim_emb   : embedding size
    # cnn_dim   : convolution dimension (default:256)
    # bltnk_dim : convolution bottleneck dimension (default:64)
    # att_dim   : attention dimension (default:64)
    # layers    : number of layers (default:12)
    # keep_prob : percentage that is kept when using dropout
    # train     : boolean, "True"" during Training else "False"


    with tf.variable_scope("input"):
        tweets = placeholder(tf.int32, (None, None), data[0], 'tweets')
        labels = placeholder(tf.float32, (None,), data[1], 'labels')
        train = placeholder(tf.bool, (), train, 'training')

    with tf.variable_scope('embed'):
        # (b, t) -> (b, t, dim_emb)
        embed_mtrx = tf.get_variable(name="embed_mtrx", shape=[dim_voc, dim_emb])
        embed = tf.gather(embed_mtrx, tweets)
        x = tf.layers.dropout(embed, keep_prob, training=train)

    with tf.variable_scope('CNN'):
        # (b, t, dim_emb) -> (b,t,units)
        # bottleneck convolution (google: lenet)
        x = conv1D(x, cnn_dim)
        for _ in range(layers):
            x = btlnk_residual(x, btlnk_dim, cnn_dim, train, keep_prob)

    with tf.variable_scope('attention'):
        # (b,att_dim)
        # normal bd-attention
        x = attention(x, att_dim)

    step = tf.train.get_or_create_global_step()

    with tf.variable_scope('logits'):
        logits = tf.squeeze(tf.layers.dense(x, 1), axis=-1)  # (b,)

    with tf.variable_scope('cross_entropy'):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)

    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(cross_entropy)

    with tf.variable_scope('pred'):
        cond = tf.greater(logits, 0)
        pred = tf.cast(cond, tf.float32)

    with tf.variable_scope('acc'):
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(labels, tf.float32), pred), tf.float32))

    with tf.variable_scope('train_step'):
        train_step = tf.train.AdamOptimizer().minimize(loss, step)

    return dict(step=step,
                tweets=tweets,
                labels=labels,
                acc=acc,
                logits=logits,
                pred=pred,
                train=train,
                embed_mtrx=embed_mtrx,
                loss=loss,
                train_step=train_step)
