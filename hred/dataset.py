import tensorflow as tf
from collections import namedtuple
from tensorflow.python.ops import lookup_ops


PAD = '<pad>'
SOS = '<s>'
EOS = '</s>'
UNK = '<unk>'


class BatchedInput(namedtuple("BatchedInput",
                              ("initializer",
                               "sources",
                               "targets_in",
                               "targets_out",
                               "src_lengths",
                               "tgt_lengths"))):
    pass


def _pad_1d_tensor(tensor, max_length, dtype=tf.int32):
    return tf.concat(
        [tensor, tf.zeros((max_length - tf.size(tensor), ), dtype=dtype)], 0)


def get_vocab_size(vocab_file):
    with open(vocab_file, 'r', encoding='utf-8') as f:
        num_vocab = len([line.strip() for line in f])
    return num_vocab + 4


def get_vocab_table(vocab_file, reverse=False):
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocabs = [PAD, SOS, EOS, UNK] + [line.strip() for line in f]
    if not reverse:
        vocab_table = lookup_ops.index_table_from_tensor(
            vocabs, default_value=3)
    else:
        vocab_table = lookup_ops.index_to_string_table_from_tensor(
            vocabs, default_value=UNK)

    return vocab_table


def get_iterator(corpus_file, vocab_table, batch_size,
                 num_threads=4, output_buffer_size=None):
    if output_buffer_size is None:
        output_buffer_size = batch_size * 1000

    sos_id = tf.cast(vocab_table.lookup(tf.constant(SOS)), tf.int32)
    eos_id = tf.cast(vocab_table.lookup(tf.constant(EOS)), tf.int32)

    # get dataset
    def _add_token_and_split(*sentences):
        sources = [tf.concat((l, [eos_id]), 0) for l in sentences[:-1]]
        targets_in = [tf.concat(([sos_id], l), 0) for l in sentences[1:]]
        targets_out = [tf.concat((l, [eos_id]), 0) for l in sentences[1:]]

        src_lengths = tf.stack([tf.size(x) for x in sources])
        tgt_lengths = tf.stack([tf.size(x) for x in targets_in])
        src_max_length = tf.reduce_max(src_lengths)
        tgt_max_length = tf.reduce_max(tgt_lengths)

        sources = tf.stack([_pad_1d_tensor(s, src_max_length) for s in sources])
        targets_in = tf.stack(
            [_pad_1d_tensor(t, tgt_max_length) for t in targets_in])
        targets_out = tf.stack(
            [_pad_1d_tensor(t, tgt_max_length) for t in targets_out])
        return sources, targets_in, targets_out, src_lengths, tgt_lengths

    dataset = tf.contrib.data.TextLineDataset(corpus_file)
    dataset = dataset.map(
        lambda line: tf.string_split([line], delimiter='\t').values,
        num_threads=num_threads, output_buffer_size=output_buffer_size)
    dataset = dataset.map(
        lambda sents: tuple(
            tf.cast(vocab_table.lookup(tf.string_split([sents[i]]).values),
                    tf.int32)
            for i in range(3)),
        num_threads=num_threads, output_buffer_size=output_buffer_size)
    dataset = dataset.map(
        _add_token_and_split,
        num_threads=num_threads, output_buffer_size=output_buffer_size)

    batched_dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=(tf.TensorShape([None, None]),
                       tf.TensorShape([None, None]),
                       tf.TensorShape([None, None]),
                       tf.TensorShape([None]),
                       tf.TensorShape([None])),
        padding_values=(0, 0, 0, 0, 0)
    )
    iterator = batched_dataset.make_initializable_iterator()
    sources, targets_in, targets_out, src_lengths, tgt_lengths = \
        iterator.get_next()

    return BatchedInput(initializer=iterator.initializer,
                        sources=sources,
                        targets_in=targets_in,
                        targets_out=targets_out,
                        src_lengths=src_lengths,
                        tgt_lengths=tgt_lengths)


def to_single_string(strings):
    strings = [st.decode('utf-8') for st in strings]
    strings = [st for st in strings if st not in {EOS, SOS, PAD}]
    return ' '.join(strings)
