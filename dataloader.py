import tensorflow as tf
from configs import MAX_LEN, BATCH_SIZE, AUTO
from tokenizer_configs import bert_tokenizer, random_selector, _PLAIN_TOKEN, _HUMOR_TOKEN, _ROMANCE_TOKEN, _CLICK_TOKEN, _MASK_TOKEN

def read_tfrec(example):
    tfrec_format = {
        "text": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    text = tf.io.decode_raw(example['text'], tf.int64)
    text = text[:MAX_LEN]
    text = tf.reshape(text, (MAX_LEN,))
    return text


def sample_keys(input, guide = False):
    mask_ratio = tf.random.uniform((1,), 0.1, 0.3)
    max_len = tf.math.reduce_sum(tf.cast(tf.cast(input, tf.bool), tf.int32))
    mask = tf.random.categorical(tf.math.log(tf.transpose([mask_ratio, (1-mask_ratio)])), max_len-2, dtype = tf.int64)
    # mask = tf.reshape(mask, -1)
    input = tf.cast(input, tf.int64)
    result = tf.expand_dims(input[1:max_len-1],0)
    result = tf.boolean_mask(result, mask)
    # result = tf.random.shuffle(result)
    result = tf.concat([[101], result, [102], tf.zeros(MAX_LEN-max_len + tf.math.reduce_sum(tf.cast(1-mask, tf.int32)), dtype = tf.int64)], axis = 0)
    result = tf.reshape(result, (MAX_LEN,))
    if guide == True:
        result = (result, result)
    return (result, tf.reshape(input, (MAX_LEN,)))


def add_noise(input, guide = False):
    mask_ratio = tf.random.uniform((1,), 0.1, 1)
    max_len = tf.cast(tf.cast(input, tf.bool), tf.int64)
    mask = tf.random.categorical(tf.math.log(tf.transpose([mask_ratio, (1-mask_ratio)])), MAX_LEN, dtype = tf.int64)
    # mask = tf.reshape(mask, -1)
    unks = mask * tf.cast(tf.fill((1,MAX_LEN), _MASK_TOKEN), dtype = tf.int64)
    masked = input*(1-mask)
    out = unks + masked
    out = out * max_len
    return tf.reshape(out, (MAX_LEN,)), tf.reshape(input, (MAX_LEN,))


def get_ds(filenames, process = None, loader = read_tfrec, batch_size = BATCH_SIZE, cache = True, repeat = False, shuffle = False, drop_remainder = True, AUTO = -1,):
    ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    if cache:
        ds = ds.cache()
    if repeat:
        ds = ds.repeat()
    if shuffle:
        ds = ds.shuffle(1024 * 2)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)

    ds = ds.map(read_tfrec, num_parallel_calls=AUTO)

    if process is not None:
        ds = ds.map(process, num_parallel_calls=AUTO)
    if batch_size is not None:
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(AUTO)

    return ds

def get_ds_pd(df, process, batch_size = BATCH_SIZE, cache = True, repeat = False, shuffle = False, drop_remainder = True):
    ds = tf.data.Dataset.from_tensor_slices(df)
    if cache:
        ds = ds.cache()
    if repeat:
        ds = ds.repeat()
    if shuffle:
        ds = ds.shuffle(1024 * 2)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)

    if process is not None:
        ds = ds.map(process, num_parallel_calls=AUTO)
    if batch_size is not None:
        ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(AUTO)

    return ds

def tf_bert_sampler(text, guide = True, offset=True, style_token = _PLAIN_TOKEN, style = True):
    text_token = bert_tokenizer.tokenize(text)
    text_token = text_token[:MAX_LEN]
    selected = random_selector.get_selection_mask(text_token, axis=1)
    result = tf.ragged.boolean_mask(text_token,selected)[0]
    shape = tf.math.reduce_sum(tf.cast(selected, tf.int32))
    a = tf.random.shuffle(tf.range(shape))
    b = tf.reshape(a, (shape, 1))
    result = tf.gather_nd(result, b).merge_dims(0,-1)
    result = result[:MAX_LEN]
    max_len = tf.math.reduce_sum(tf.cast(tf.cast(result, tf.bool), tf.int32))
    if style_token == 'random':
        pool = tf.constant([_CLICK_TOKEN, _HUMOR_TOKEN, _ROMANCE_TOKEN])
        ind = tf.random.uniform(shape=(), minval=0, maxval=3, dtype = tf.int64)
        style_token = tf.gather(pool,ind)

    result = tf.concat([[style_token]*style,[101], result, [102], tf.zeros(MAX_LEN-max_len-3+(1-style), dtype=tf.int64)], axis = 0)

    target = text_token.merge_dims(0,-1)
    target = target[:MAX_LEN-2]
    max_len_target = tf.math.reduce_sum(tf.cast(tf.cast(target, tf.bool), tf.int32))
    target = tf.concat([[101], target, [102], tf.zeros(tf.nn.relu(MAX_LEN-max_len_target-2), dtype=tf.int64)], axis = 0)

    if guide:
      if offset:
        offset = tf.concat([target[1:], [0]], axis = 0)
        return ((tf.reshape(result,(MAX_LEN,)) , tf.reshape(target, (MAX_LEN,))), tf.reshape(offset, (MAX_LEN,)))
      else:
        return ((tf.reshape(result,(MAX_LEN,)) , tf.reshape(target, (MAX_LEN,))), tf.reshape(target, (MAX_LEN,)))
    else:
        return (tf.reshape(result, (MAX_LEN,)), tf.reshape(target, (MAX_LEN,)))

    return result