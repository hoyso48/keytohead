import tensorflow as tf

class Translator(tf.Module):
  def __init__(self, tokenizer, transformer):
    self.tokenizer = tokenizer
    self.transformer = transformer

  def __call__(self, sentence, max_length=20):
    # input sentence is portuguese, hence adding the start and end token
    # assert isinstance(sentence, tf.Tensor)
    # if len(sentence.shape) == 0:
    #   sentence = sentence[tf.newaxis]

    # sentence = self.tokenizer.pt.tokenize(sentence).to_tensor()

    encoder_input = sentence

    # as the target is english, the first token to the transformer should be the
    # english start token.
    # start_end = self.tokenizer([''])['input_ids']
    # print(start_end)
    # start = start_end[0][tf.newaxis]
    # end = start_end[1][tf.newaxis]
    start = [101]
    end = [102]

    # `tf.TensorArray` is required here (instead of a python list) so that the
    # dynamic-loop can be traced by `tf.function`.
    output_array = tf.TensorArray(dtype=tf.int64, size=64)
    output_array = output_array.write(0, start)
    print(output_array)

    for i in tf.range(max_length):
      output = tf.transpose(output_array.stack())
      predictions = self.transformer([encoder_input, output], training=False)

      # select the last token from the seq_len dimension
      predictions = predictions[:, i, :]  # (batch_size, 1, vocab_size)

      predicted_id = tf.argmax(predictions, axis=-1)
      print(predicted_id)

      # concatentate the predicted_id to the output which is given to the decoder
      # as its input.
      output_array = output_array.write(i+1, predicted_id)

      if predicted_id == end:
        break

    output = tf.transpose(output_array.stack())
    # output.shape (1, tokens)
    text = self.tokenizer.batch_decode(output)[0]  # shape: ()

    # tokens = tokenizer.en.lookup(output)[0]

    # `tf.function` prevents us from using the attention_weights that were
    # calculated on the last iteration of the loop. So recalculate them outside
    # the loop.
    # _, attention_weights = self.transformer([encoder_input, output[:,:-1]], training=False)

    return text#, tokens, attention_weights