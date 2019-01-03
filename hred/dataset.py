import tensorflow as tf
import numpy as np
from collections import namedtuple
from tensorflow.python.ops import lookup_ops


PAD = '<pad>'
UNK = '<unk>'
SOS = '<s>'
EOS = '</s>'


class BatchedInput(namedtuple("BatchedInput",
                              ("initializer",
                               "sources",
                               "targets_in",
                               "targets_out",
                               "src_lengths",
                               "tgt_lengths"))):
    pass


class InferInput(namedtuple("InferInput",
                            ("placeholder",
                             "sources",
                             "src_lengths"))):
    pass


def _pad_1d_tensor(tensor, max_length, dtype=tf.int32):
    return tf.concat(
        [tensor, tf.zeros((max_length - tf.size(tensor), ), dtype=dtype)], 0)


def get_vocab_size(vocab_file):
    with open(vocab_file, 'r', encoding='utf-8') as f:
        num_vocab = len([line.strip() for line in f])
    return num_vocab + 4


def get_vocab_table(vocab_file, reverse=False):
    vocabs = [PAD, UNK, SOS, EOS]
    counts = [0]*len(vocabs)
    with open(vocab_file, 'r', encoding='utf-8') as f:
        for line in f:
            vocab, count = line.strip().split('\t')
            vocabs.append(vocab)
            counts.append(int(count))

    # probability of EOS should be 1. / (average target token num + 1).
    # currently 2.75
    sum_counts = sum(counts)
    # TODO: fix here to calculate average from input data instead for hardcoded one
    eos_prob = 1. / (2.75 + 1)
    vocab_probs = np.array(counts, dtype=np.float32) * (1 - eos_prob) \
                  / sum_counts
    vocab_probs[3] = eos_prob

    if not reverse:
        vocab_table = lookup_ops.index_table_from_tensor(
            vocabs, default_value=1)
    else:
        vocab_table = lookup_ops.index_to_string_table_from_tensor(
            vocabs, default_value=UNK)

    return vocab_table, vocab_probs


def get_iterator(corpus_file, vocab_table, batch_size,
                 skip_unk_target=False,
                 num_threads=4, output_buffer_size=None):
    if output_buffer_size is None:
        output_buffer_size = batch_size * 1000
    with open(corpus_file, 'r', encoding='utf-8') as f:
        num_sentences = len(f.readline().strip().split('\t'))

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

    dataset = tf.data.TextLineDataset(corpus_file)
    dataset = dataset.shuffle(output_buffer_size)

    dataset = dataset.map(
        lambda line: tf.string_split([line], delimiter='\t').values,
        num_parallel_calls=num_threads)
    dataset = dataset.map(
        lambda sents: tuple(
            tf.cast(vocab_table.lookup(tf.string_split([sents[i]]).values),
                    tf.int32)
            for i in range(num_sentences)),
        num_parallel_calls=num_threads)

    if skip_unk_target:
        unk_idx = tf.constant(1, dtype=tf.int32)

        def _unk_in_target(*sentences):
            targets = sentences[1:]
            is_unks = [tf.reduce_any(tf.equal(t, unk_idx)) for t in targets]
            return tf.reduce_all(tf.logical_not(tf.stack(is_unks)))
        dataset = dataset.filter(_unk_in_target)

    dataset = dataset.map(
        _add_token_and_split,
        num_parallel_calls=num_threads)

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


def get_infer_inputs(vocab_table, num_sentence):
    """
    Return iterator placeholder and input tensors
    """
    eos_id = tf.cast(vocab_table.lookup(tf.constant(EOS)), tf.int32)

    input_strings = tf.placeholder(tf.string, [num_sentence])
    split_inputs = tuple(
        tf.cast(vocab_table.lookup(tf.string_split([sent])).values, tf.int32)
        for sent in tf.unstack(input_strings)
    )
    sources = [tf.concat((l, [eos_id]), 0) for l in split_inputs]
    src_lengths = tf.stack([tf.size(x) for x in sources])
    src_max_length = tf.reduce_max(src_lengths)
    sources = tf.stack([_pad_1d_tensor(s, src_max_length) for s in sources])

    # add batch dim
    sources = tf.expand_dims(sources, axis=0)
    src_lengths = tf.expand_dims(src_lengths, axis=0)

    return InferInput(placeholder=input_strings,
                      sources=sources,
                      src_lengths=src_lengths)


def to_single_string(strings):
    strings = [st.decode('utf-8') for st in strings]
    try:
        eos_index = strings.index(EOS)
        strings = strings[:eos_index]
    except ValueError:
        pass
    strings = [st for st in strings if st not in {EOS, SOS, PAD}]
    return ' '.join(strings)
