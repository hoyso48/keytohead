from tensorflow.python.ops.array_ops import zeros_like_impl
from layers import *
from tensorflow.keras.layers import *
from configs import *
#ED for encoder-decoder
#BE for bidirectional encoder

def Transformer_ED_12(d_model = 768, num_heads = 12):
    emb = TokenAndPositionEmbedding(MAX_LEN, VOCAB_SIZE, 512)
    input = L.Input((MAX_LEN,))
    target = L.Input((MAX_LEN,))
    x = emb(input)
    y = emb(target)
    mask = create_mask(target, MAX_LEN)
    mask_enc = create_pad_mask(input, MAX_LEN)
    mask_dec = create_pad_mask(target, MAX_LEN)

    x = Transformerblock(d_model = d_model, num_heads = num_heads)(x,x,x,mask_enc)
    x = Transformerblock(d_model = d_model, num_heads = num_heads)(x,x,x,mask_enc)
    x = Transformerblock(d_model = d_model, num_heads = num_heads)(x,x,x,mask_enc)
    x = Transformerblock(d_model = d_model, num_heads = num_heads)(x,x,x,mask_enc)
    x = Transformerblock(d_model = d_model, num_heads = num_heads)(x,x,x,mask_enc)
    enc_out = Transformerblock(d_model = d_model, num_heads = num_heads)(x,x,x,mask_enc)


    y = Transformerblock(d_model = d_model, num_heads = num_heads,ff=False)(y,y,y,mask)
    y = Transformerblock(d_model = d_model, num_heads = num_heads,ff=False)(y,enc_out,enc_out,mask_enc)
    y = Transformerblock(d_model = d_model, num_heads = num_heads,att=False)(y,y,y,mask_dec)
    y = Transformerblock(d_model = d_model, num_heads = num_heads,ff=False)(y,y,y,mask)
    y = Transformerblock(d_model = d_model, num_heads = num_heads,ff=False)(y,enc_out,enc_out,mask_enc)
    y = Transformerblock(d_model = d_model, num_heads = num_heads,att=False)(y,y,y,mask_dec)
    y = Transformerblock(d_model = d_model, num_heads = num_heads,ff=False)(y,y,y,mask)
    y = Transformerblock(d_model = d_model, num_heads = num_heads,ff=False)(y,enc_out,enc_out,mask_enc)
    y = Transformerblock(d_model = d_model, num_heads = num_heads,att=False)(y,y,y,mask_dec)
    y = Transformerblock(d_model = d_model, num_heads = num_heads,ff=False)(y,y,y,mask)
    y = Transformerblock(d_model = d_model, num_heads = num_heads,ff=False)(y,enc_out,enc_out,mask_enc)
    y = Transformerblock(d_model = d_model, num_heads = num_heads,att=False)(y,y,y,mask_dec)
    y = Transformerblock(d_model = d_model, num_heads = num_heads,ff=False)(y,y,y,mask)
    y = Transformerblock(d_model = d_model, num_heads = num_heads,ff=False)(y,enc_out,enc_out,mask_enc)
    y = Transformerblock(d_model = d_model, num_heads = num_heads,att=False)(y,y,y,mask_dec)
    y = Transformerblock(d_model = d_model, num_heads = num_heads,ff=False)(y,y,y,mask)
    y = Transformerblock(d_model = d_model, num_heads = num_heads,ff=False)(y,enc_out,enc_out,mask_enc)
    dec_out = Transformerblock(d_model = d_model, num_heads = num_heads,att=False)(y,y,y,mask_dec)
    out = L.Dense(VOCAB_SIZE)(dec_out)

    model = tf.keras.models.Model([input,target], out)
    model.summary()
    return model

def AttConv1D_ED_12(d_model = 768, num_heads = 12, kernel_size = 8):
    emb = TokenAndPositionEmbedding(MAX_LEN, VOCAB_SIZE, 512)
    input = L.Input((MAX_LEN,))
    target = L.Input((MAX_LEN,))
    x = emb(input)
    y = emb(target)
    mask = create_mask(target, MAX_LEN)
    mask_enc = create_pad_mask(input, MAX_LEN)
    # mask_dec = create_pad_mask(target)

    x = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(x,x,x,mask_enc)
    x = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(x,x,x,mask_enc)
    x = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(x,x,x,mask_enc)
    x = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(x,x,x,mask_enc)
    x = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(x,x,x,mask_enc)
    enc_out = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(x,x,x,mask_enc)


    y = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(y,y,y,mask)
    y = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(y,enc_out,enc_out,mask_enc)
    y = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(y,y,y,mask)
    y = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(y,enc_out,enc_out,mask_enc)
    y = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(y,y,y,mask)
    y = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(y,enc_out,enc_out,mask_enc)
    y = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(y,y,y,mask)
    y = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(y,enc_out,enc_out,mask_enc)
    y = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(y,y,y,mask)
    y = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(y,enc_out,enc_out,mask_enc)
    y = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(y,y,y,mask)
    dec_out = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(y,enc_out,enc_out,mask_enc)

    out = L.Dense(VOCAB_SIZE)(dec_out)

    model = tf.keras.models.Model([input,target], out)
    model.summary()
    return model
    
def Transformer_BE_12(d_model = 768, num_heads = 12):
    emb = BertEmbedding(MAX_LEN, VOCAB_SIZE, 768)
    input = L.Input((MAX_LEN,))
    target = L.Input((MAX_LEN,))
    z = Concatenate(axis = 1)([input,target])
    mask = create_bert_mask(z, MAX_LEN)
    z = emb(z)

    z = Transformerblock(d_model = d_model, num_heads = num_heads)(z,z,z)
    z = Transformerblock(d_model = d_model, num_heads = num_heads)(z,z,z)
    z = Transformerblock(d_model = d_model, num_heads = num_heads)(z,z,z)
    z = Transformerblock(d_model = d_model, num_heads = num_heads)(z,z,z)
    z = Transformerblock(d_model = d_model, num_heads = num_heads)(z,z,z)
    z = Transformerblock(d_model = d_model, num_heads = num_heads)(z,z,z)
    z = Transformerblock(d_model = d_model, num_heads = num_heads)(z,z,z)
    z = Transformerblock(d_model = d_model, num_heads = num_heads)(z,z,z)
    z = Transformerblock(d_model = d_model, num_heads = num_heads)(z,z,z)
    z = Transformerblock(d_model = d_model, num_heads = num_heads)(z,z,z)
    z = Transformerblock(d_model = d_model, num_heads = num_heads)(z,z,z)
    enc_out = Transformerblock(d_model = d_model, num_heads = num_heads)(z,z,z)

    out = L.Dense(VOCAB_SIZE)(enc_out)

    model = tf.keras.models.Model([input,target], out)
    model.summary()
    return model

def AttConv1D_BE_12(d_model = 768, num_heads = 12, kernel_size = 8):
    emb = BertEmbedding(MAX_LEN, VOCAB_SIZE, 768)
    input = L.Input((MAX_LEN,))
    target = L.Input((MAX_LEN,))
    z = Concatenate(axis = 1)([input,target])
    mask = create_bert_mask(z, MAX_LEN)
    z = emb(z)

    z = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(z,z,z,mask)
    z = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(z,z,z,mask)
    z = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(z,z,z,mask)
    z = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(z,z,z,mask)
    z = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(z,z,z,mask)
    z = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(z,z,z,mask)
    z = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(z,z,z,mask)
    z = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(z,z,z,mask)
    z = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(z,z,z,mask)
    z = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(z,z,z,mask)
    z = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(z,z,z,mask)
    enc_out = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(z,z,z,mask)[:,MAX_LEN:2*MAX_LEN,:]

    out = L.Dense(VOCAB_SIZE)(enc_out)

    model = tf.keras.models.Model([input,target], out)
    model.summary()
    return model

def StyleAttConv1D_BE_12(d_model = 768, num_heads = 12, kernel_size = 8):
    emb = BertEmbedding(MAX_LEN, VOCAB_SIZE, 768)
    input = L.Input((MAX_LEN,))
    target = L.Input((MAX_LEN,))
    z = Concatenate(axis = 1)([input,target])
    mask = create_bert_mask(z, MAX_LEN)
    z = emb(z)
    style = input[0][0]

    z = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(z,z,z,mask)
    z = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(z,z,z,mask)
    z = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(z,z,z,mask)
    z = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(z,z,z,mask)
    z = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(z,z,z,mask)
    z = AttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(z,z,z,mask)
    z = StyleAttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(style,z,z,z,mask)
    z = StyleAttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(style,z,z,z,mask)
    z = StyleAttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(style,z,z,z,mask)
    z = StyleAttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(style,z,z,z,mask)
    z = StyleAttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(style,z,z,z,mask)
    enc_out = StyleAttConv1D(d_model = d_model, num_heads = num_heads, kernel_size = 8)(style,z,z,z,mask)[:,MAX_LEN:2*MAX_LEN,:]

    out = L.Dense(VOCAB_SIZE)(enc_out)

    model = tf.keras.models.Model([input,target], out)
    model.summary()
    return model

def BERT(bert_layers):
    tok_emb = bert_layers.layers[1]
    seg_emb = bert_layers.layers[2]
    pos_emb = bert_layers.layers[3]
    transformer_layers = bert_layers.layers[10:-2]
    input_ids = Input(shape=(MAX_LEN,), name='input_ids', dtype = 'int64')
    target_ids = Input(shape=(MAX_LEN,), name='target_ids', dtype = 'int64')
    inputs = Concatenate(axis = 1)([input_ids, target_ids])

    tok = tok_emb(inputs)
    pos = pos_emb(tok)
    seg = seg_emb(tf.concat([tf.zeros(MAX_LEN//2), tf.ones(MAX_LEN//2)], axis = 1))
    x = tok + pos + seg

    mask = create_bert_mask(inputs, MAX_LEN)
    # for layer in transformer_layers:
    #   x = layer([x, mask])
    x = transformer_layers[0]([x,mask])
    x = transformer_layers[1]([x,mask])
    x = transformer_layers[2]([x,mask])
    x = transformer_layers[3]([x,mask])
    x = transformer_layers[4]([x,mask])
    x = transformer_layers[5]([x,mask])
    x = transformer_layers[6]([x,mask])
    x = transformer_layers[7]([x,mask])
    x = transformer_layers[8]([x,mask])
    x = transformer_layers[9]([x,mask])
    x = transformer_layers[10]([x,mask])
    x = x[:,MAX_LEN:2*MAX_LEN,:]
    outputs = Dense(units=VOCAB_SIZE, name='issue')(x)
    model = tf.keras.models.Model(inputs=[input_ids,target_ids], outputs=outputs, name='BERT')
    model.summary()
    return model