import tensorflow as tf
import numpy as np
import warnings

from .dataset import get_vocab_table, get_iterator
from .model import HRED


def get_model(hparams, mode, graph):
    if mode == tf.contrib.learn.ModeKeys.TRAIN:
        corpus_path = hparams.train_path
        batch_size = hparams.batch_size
    elif mode == tf.contrib.learn.ModeKeys.EVAL:
        corpus_path = hparams.val_path
        batch_size = hparams.val_batch_size
    else:
        raise ValueError('Inappropriate mode for get_model.')

    with tf.device('/cpu:0'):
        vocab_table, vocab_probs = get_vocab_table(hparams.vocab_path)
        reverse_vocab_table, _ = get_vocab_table(hparams.vocab_path, reverse=True)
        iterator = get_iterator(corpus_path, vocab_table, batch_size)
    model = HRED(iterator, vocab_table, reverse_vocab_table,
                 vocab_probs, hparams, mode)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config, graph=graph)
    sess.run(tf.tables_initializer())

    return model, iterator, sess


def _single_cell(cell_type, num_units, dropout_keep_prob, residual):
    if cell_type.lower() == 'lstm':
        cell = tf.contrib.rnn.LSTMCell(num_units=num_units)
    elif cell_type.lower() == 'gru':
        cell = tf.contrib.rnn.GRUCell(num_units=num_units)
    elif cell_type.lower() == 'layernormlstm':
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=num_units)
    else:
        raise ValueError('Inappropriate cell type {}'.format(cell_type))

    cell = tf.contrib.rnn.DropoutWrapper(
        cell, input_keep_prob=dropout_keep_prob)
    if residual:
        cell = tf.contrib.rnn.ResidualWrapper(cell)
    return cell


def create_rnn_cell(cell_type, num_layers, num_units,
                    dropout_keep_prob, num_residual_layers):
    """
    Create RNN cell.
    :param cell_type: Type of cell, among ['lstm', 'gru', 'layernormlstm']
    :param num_layers: Number of stacked rnn layers
    :param num_units: Size of hidden dimension for each cell
    :param dropout_keep_prob: Dropout keep prob
    :param num_residual_layers: number of residual layers.
        If num_residual_layers < num_layers, residual cells
        are stacked on the top part of layers.
    :return: Created RNN Cell
    """
    rnn_cells = \
        [_single_cell(cell_type, num_units, dropout_keep_prob,
                      layer_num + num_residual_layers > num_layers)
         for layer_num in range(1, num_layers+1)]

    if len(rnn_cells) == 0:
        return rnn_cells[0]
    else:
        return tf.contrib.rnn.MultiRNNCell(rnn_cells)


def create_word_embedding(num_vocab, embedding_dim, name,
                          pretrained_word_matrix=None, trainable=True):
    """
    Create Word embedding variable.
    :param num_vocab: Number of vocabs
    :param embedding_dim: Dimension of word embedding
    :param name: Name of variable
    :param pretrained_word_matrix: 2D numpy array.
      If not None, initialize embedding from this array.
    :param trainable: Boolean for indicating if this variable is trainable.
    :return: Created word embedding
    """
    vocab_shape = (num_vocab, embedding_dim)
    initializer = None
    if pretrained_word_matrix:
        if type(pretrained_word_matrix) == str:
            with open(pretrained_word_matrix, 'rb') as f:
                pretrained_word_matrix = np.load(pretrained_word_matrix)
        if pretrained_word_matrix.shape == vocab_shape:
            initializer = pretrained_word_matrix
        else:
            warnings.warn(
                ('Shape of pretrained word matrix {} Does not match '
                 'the size of expected shape {}, thus initialize from scratch')
                .format(pretrained_word_matrix.shape, vocab_shape))

    return tf.get_variable(
        name,
        shape=vocab_shape,
        dtype=tf.float32,
        initializer=initializer,
        trainable=trainable)


def create_attention_mechanism(attention_option, num_units,
                               memory, source_length):
    """Create attention mechanism."""
    if attention_option.lower() == "bahdanau":
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units, memory,
            memory_sequence_length=source_length,
            normalize=True)
    elif attention_option.lower() == "luong":
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units, memory,
            memory_sequence_length=source_length,
            scale=True)
    else:
        raise ValueError('Unknown attention mechanism {}'
                         .format(attention_option))

    return attention_mechanism
