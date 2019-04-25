import tensorflow as tf
from util_np import vpack, np, sample
from util_io import load_txt

def profile(sess, wtr, run, feed_dict= None, prerun= 3, tag= 'flow'):
    for _ in range(prerun): sess.run(run, feed_dict)
    meta = tf.RunMetadata()
    sess.run(run, feed_dict, tf.RunOptions(trace_level= tf.RunOptions.FULL_TRACE), meta)
    wtr.add_run_metadata(meta, tag)


def pipe(*args, prefetch=1, repeat=-1, name='pipe', **kwargs):
    """see `tf.data.Dataset.from_generator`."""
    with tf.variable_scope(name):
        return tf.data.Dataset.from_generator(*args, **kwargs) \
                              .repeat(repeat) \
                              .prefetch(prefetch) \
                              .make_one_shot_iterator() \
                              .get_next()


def placeholder(dtype, shape, x= None, name= None):
    """returns a placeholder with `dtype` and `shape`.

    if tensor `x` is given, converts and uses it as default.

    """
    if x is None: return tf.placeholder(dtype, shape, name)
    try:
        x = tf.convert_to_tensor(x, dtype)
    except ValueError:
        x = tf.cast(x, dtype)
    return tf.placeholder_with_default(x, shape, name)


def batch(size, path_txt, path_lbl, vocab, seed, eos, max_len):
    tweets = tuple(load_txt(path_txt))
    lbl = np.load(path_lbl)
    batch, b_lbl = [], []
    for i in sample(len(tweets), seed):
        if size == len(batch):
            batch = vpack(batch, (size, max(map(len, batch))), eos, np.int32)
            output = batch, np.asarray(b_lbl, dtype=np.int32)
            yield output
            batch, b_lbl = [], []
        inst = vocab.sample_encode_as_ids(tweets[i], -1, 0.1)
        if len(inst)<=max_len:
            batch.append(inst)
            b_lbl.append(lbl[i])


def conv1D(inpt, units, kernel_size=3, stride=1, padding='same', activation=tf.nn.relu):
    output = tf.layers.conv1d(inpt,
                              filters=units,
                              kernel_size=kernel_size,
                              strides=stride,
                              padding=padding,
                              activation=activation)
    return output


def btlnk_residual(inpt, btlnk_dim, cnn_dim, train, keep_prob=None):
    """bottleneck convolution, preset is the standard version"""
    x = conv1D(inpt, btlnk_dim, kernel_size=1)
    x = conv1D(x, btlnk_dim, kernel_size=3)
    x = conv1D(x, cnn_dim, kernel_size=1, activation=None)
    x = tf.layers.batch_normalization(x, training=train)
    x = tf.nn.relu(tf.add(inpt, x))
    if keep_prob != None:
        x = tf.layers.dropout(x, 1-keep_prob, training=train)
    return x


def attention(values, att_dim, query=None):
    # values: (b,t,c)
    # query : empty or (1,c)

    if query==None:
        h_val = int(values.get_shape()[-1])
        query = tf.get_variable("query", [1, h_val]) #(1,c)

    Wq = tf.layers.dense(query, att_dim, use_bias=False) #(1,dim)
    Wv = tf.layers.dense(values, att_dim, use_bias=False) # (b,t,dim)
    e = tf.layers.dense(tf.nn.tanh(Wq+Wv), 1, use_bias=False) #broadcasting (b,t,dim_q+dim_v) -> (b,t,1)
    align = tf.nn.softmax(tf.squeeze(e, axis=-1)) # (b,t)
    scores = tf.expand_dims(align, 1) # (b,1,t)
    context = tf.squeeze(scores @ values, axis=1) # (b,1,t)@(b,t,c) -> (b,1,c)
    return context
