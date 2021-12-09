import tensorflow as tf
import tensorflow.keras.layers as L
import numpy as np
from tokenizer_configs import _PLAIN_TOKEN, _HUMOR_TOKEN, _ROMANCE_TOKEN, _CLICK_TOKEN

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def create_bidirectional_lookahead_mask(max_len):
  prefix_mask = tf.transpose(tf.repeat(tf.concat([tf.ones((max_len,1)), tf.zeros((max_len,1))],axis = 0), max_len*2, axis = -1))
  causal_mask = 1-create_look_ahead_mask(max_len*2)
  return tf.maximum(prefix_mask, causal_mask)[tf.newaxis, :,:]

def create_pad_mask(seq, max_len):
  seq = 1-tf.cast(tf.math.equal(seq, 0), tf.float32)#[:,tf.newaxis]
#   pad = tf.ones((1, 1, max_len))
#   mask = tf.concat([seq, pad], axis = -1)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:,tf.newaxis,:]#tf.repeat(seq, max_len, axis = 1)

def create_bert_mask(seq, max_len):
  pad_mask = create_pad_mask(seq, max_len)
  bert_mask = create_bidirectional_lookahead_mask(max_len)
  return tf.minimum(pad_mask, bert_mask)

def create_mask(seq, max_len):
  pad_mask = create_pad_mask(seq, max_len)
  lookahead_mask = (1-create_look_ahead_mask(max_len))[tf.newaxis,:,:]
  return tf.maximum(pad_mask, lookahead_mask)


def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len, vocab_size, embedding_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.pos_emb = tf.keras.layers.Embedding(max_len, embedding_dim)

    def call(self, x):
        max_len = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=max_len, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions#tf.math.sqrt(tf.cast(512, tf.float32))*x + positions

class BertEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_len, vocab_size, embedding_dim):
        super(BertEmbedding, self).__init__()
        self.token_emb = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.pos_emb = tf.keras.layers.Embedding(max_len*2, embedding_dim)
        self.seg_emb = tf.keras.layers.Embedding(2, embedding_dim)
        self.max_len = max_len

    def call(self, x):
        positions = tf.range(start=0, limit=self.max_len*2, delta=1)
        positions = self.pos_emb(positions)
        segments = tf.concat([tf.zeros(self.max_len), tf.ones(self.max_len)], axis = 0)
        segments = self.seg_emb(segments)
        x = self.token_emb(x)
        return x + positions + segments#tf.math.sqrt(tf.cast(512, tf.float32))*x + positions

def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += ((1-mask) * -1e9) #0 attention score for mask=0

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
#   attention_weights = tf.nn.sigmoid(scaled_attention_logits)
  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output#, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, q, k, v, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='gelu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

class Transformerblock(tf.keras.layers.Layer):
    def __init__(self, ff=True, att=True, d_model=512, num_heads=8, dff=2048, rate=0.1, center = True):
        super(Transformerblock, self).__init__()
        self.att = att
        if att:
            self.attention = L.MultiHeadAttention(num_heads = num_heads, key_dim = d_model//num_heads, value_dim = d_model//num_heads)
            self.layernorm_att = L.LayerNormalization(epsilon = 1e-6, center = center)
            self.dropout_att = L.Dropout(rate)
        self.ff = ff
        if ff:
            self.ffn = point_wise_feed_forward_network(d_model = d_model, dff = dff)
            self.layernorm_ff = L.LayerNormalization(epsilon = 1e-6, center = center)
            self.dropout_ff = L.Dropout(rate)
    def call(self, q, k, v, mask=None):
        x = q
        if self.att:
            x = self.attention(q,k,v,mask)
            x = self.dropout_att(x)
            x = self.layernorm_att(x) + k
        if self.ff:
            x = self.ffn(x)
            x = self.dropout_ff(x)
            x = self.layernorm_ff(x) + x
        
        return x

class AttConv1D(tf.keras.layers.Layer):
    def __init__(self, ff=True, att=True, d_model=768, num_heads=8, kernel_size = 5, rate=0.1, center = True):
        super(AttConv1D, self).__init__()
        self.attention = L.MultiHeadAttention(num_heads = num_heads, key_dim = d_model//num_heads, value_dim = d_model//num_heads)
        self.layernorm_att = L.LayerNormalization(epsilon = 1e-6, center = center)
        self.dropout_att = L.Dropout(rate)
        self.conv1 = L.SeparableConv1D(d_model, kernel_size, padding='causal')
        self.dropout_conv1 = L.Dropout(rate)
        self.bn1 = L.LayerNormalization(epsilon = 1e-6, center = center)

    def call(self, q, k, v, mask=None):
        x = self.attention(q,k,v,mask)
        x = self.dropout_att(x)
        att_out = self.layernorm_att(x) + q

        x = self.conv1(q)
        x = self.bn1(x)
        x = L.Activation('swish')(x)
        x = self.dropout_conv1(x)
        conv1_out = x + q
        
        return conv1_out + att_out

class StyleLayerNormalization(tf.keras.layers.Layer):
    def __init__(self, center = True):
        super(StyleLayerNormalization, self).__init__()
        self.plain = L.LayerNormalization(epsilon = 1e-6, center = center)
        self.click = L.LayerNormalization(epsilon = 1e-6, center = center)
        self.humor = L.LayerNormalization(epsilon = 1e-6, center = center)
        self.romance = L.LayerNormalization(epsilon = 1e-6, center = center)

    def call(self, input, style):
        if style == _CLICK_TOKEN:
            return self.click(input)
        elif style == _HUMOR_TOKEN:
            return self.humor(input)
        elif style == _ROMANCE_TOKEN:
            return self.romance(input)
        else:
            return self.plain(input)

class StyleAttConv1D(tf.keras.layers.Layer):
    def __init__(self, ff=True, att=True, d_model=768, num_heads=8, kernel_size = 5, rate=0.1):
        super(StyleAttConv1D, self).__init__()
        self.attention = L.MultiHeadAttention(num_heads = num_heads, key_dim = d_model//num_heads, value_dim = d_model//num_heads)
        self.layernorm_att = StyleLayerNormalization()
        self.dropout_att = L.Dropout(rate)
        self.conv1 = L.SeparableConv1D(d_model, kernel_size, padding='causal')
        self.dropout_conv1 = L.Dropout(rate)
        self.bn1 = StyleLayerNormalization()

    def call(self, style, q, k, v, mask=None):
        x = self.attention(q,k,v,mask)
        x = self.dropout_att(x)
        att_out = self.layernorm_att(x, style) + q

        x = self.conv1(q)
        x = self.bn1(x, style)
        x = L.Activation('swish')(x)
        x = self.dropout_conv1(x)
        conv1_out = x + q
        
        return conv1_out + att_out