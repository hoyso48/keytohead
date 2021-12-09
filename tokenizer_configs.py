from transformers import BertTokenizer, TFDistilBertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
import tensorflow as tf
import tensorflow_text as text

_VOCAB = list(tokenizer.vocab.keys())
_START_TOKEN = _VOCAB.index("[CLS]")
_END_TOKEN = _VOCAB.index("[SEP]")
_MASK_TOKEN = _VOCAB.index("[MASK]")
_UNK_TOKEN = _VOCAB.index("[UNK]")
_PLAIN_TOKEN = _VOCAB.index("[unused1]")
_CLICK_TOKEN = _VOCAB.index("[unused2]")
_ROMANCE_TOKEN = _VOCAB.index("[unused3]")
_HUMOR_TOKEN = _VOCAB.index("[unused4]")

lookup_table = tf.lookup.StaticVocabularyTable(
    tf.lookup.KeyValueTensorInitializer(
      keys=_VOCAB,
      key_dtype=tf.string,
      values=tf.range(
          tf.size(_VOCAB, out_type=tf.int64), dtype=tf.int64),
      value_dtype=tf.int64),
      num_oov_buckets=1
)

random_selector = text.RandomItemSelector(
    max_selections_per_batch=20,
    selection_rate=0.5,
    unselectable_ids=[_START_TOKEN, _END_TOKEN, _UNK_TOKEN]
)

bert_tokenizer = text.BertTokenizer(lookup_table, token_out_type=tf.int64)