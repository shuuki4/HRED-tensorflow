import tensorflow as tf
import tensorflow.contrib.seq2seq as s2s
from tensorflow.python.layers import core as layers_core
from easydict import EasyDict as edict

from . import model_helper
from . import dataset
from .ops import dynamic_decode_with_concat


class HRED:
    """
    HRED Tensorflow model with prev-input attention.
    WARNING: Currently assumes fixed-size number of sentences over data.
    """
    def __init__(self, inputs, vocab_table, reverse_vocab_table,
                 vocab_probs, hparams, mode):
        self.inputs = inputs
        self.hparams = hparams
        self.vocab_table = vocab_table
        self.reverse_vocab_table = reverse_vocab_table
        self.vocab_probs = vocab_probs
        self.mode = mode
        self.global_step = tf.Variable(0, trainable=False)

        self._batch_size = tf.shape(inputs.sources)[0]
        self._num_sentence = tf.shape(inputs.sources)[1]
        self._build_graph(inputs)

        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess, log_loss=False, log_inference=False, summary=False):
        assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
        to_runs = {'train_op': self.train_op}
        if log_loss:
            to_runs['loss'] = self.loss
        if log_inference:
            to_runs.update({
                'inferences': self.inference_strings,
                'sources': self.source_strings,
                'targets': self.target_strings
            })
        if summary:
            to_runs['summary'] = self.summary_op

        return edict(sess.run(to_runs))

    def eval(self, sess):
        return edict(sess.run({
            'loss': self.loss,
            'batch_size': self._batch_size
        }))

    def inference_for_train(self, sess):
        assert self.mode != tf.contrib.learn.ModeKeys.INFER
        return edict(sess.run({
            'sources': self.source_strings,
            'targets': self.target_strings,
            'inferences': self.inference_strings
        }))

    def inference(self, sess, feed_dict):
        assert self.mode == tf.contrib.learn.ModeKeys.INFER
        return edict(sess.run({
            'inferences': self.inference_strings,
            'scores': self.scores
        }, feed_dict=feed_dict))

    def _build_graph(self, inputs):
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self._dropout_keep_prob = self.hparams.dropout_keep_prob
        else:
            self._dropout_keep_prob = 1.0

        with tf.name_scope('sentence_encoder'), \
             tf.variable_scope('sentence_encoder'):
            sentence_encoder_outputs, sentence_encoder_states = \
                self._build_sentence_encoder(inputs)
        with tf.name_scope('context_encoder'),\
             tf.variable_scope('context_encoder'):
            context_encoder_outputs, context_encoder_state = \
                self._build_context_encoder(sentence_encoder_states)
        with tf.name_scope('sentence_decoder'),\
             tf.variable_scope('sentence_decoder'):
            sentence_decoder_outputs = self._build_sentence_decoder(
                inputs, context_encoder_outputs,
                sentence_encoder_states, sentence_encoder_outputs)
            sentence_decoder_logit = sentence_decoder_outputs[0]
            sentence_decoder_sample_ids = sentence_decoder_outputs[2]

        with tf.name_scope('loss'):
            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                #logits = tf.reshape(
                #    sentence_decoder_logit,
                #    tf.stack([self._batch_size, self._num_sentence,
                #             -1, self.hparams.num_vocab]))
                #self.inference_ids = tf.argmax(logits, axis=-1,
                #                               name="inference_ids")

                rnn_outputs = tf.reshape(
                    sentence_decoder_logit, [-1, self.hparams.num_rnn_units])
                real_logits = tf.reshape(
                    self.output_layer(rnn_outputs),
                    tf.stack([self._batch_size, self._num_sentence,
                              -1, self.hparams.num_vocab]))
                self.inference_ids = tf.argmax(real_logits,
                                               axis=-1, name="inference_ids")
                loss = self._build_loss(
                    inputs,
                    sentence_decoder_logit,
                    tf.transpose(self.output_layer.kernel),
                    self.output_layer.bias
                )
                self.loss = loss
            else:
                self.scores = sentence_decoder_logit
                self.loss = tf.constant(0.)

        with tf.name_scope('optimizer'):
            if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
                step, lr = self._build_optimizer(self.loss)
            else:
                step = tf.no_op()
                lr = tf.constant(self.hparams.initial_learning_rate)

        with tf.name_scope('summary'):
            self.summary_op = tf.summary.merge_all()

        with tf.name_scope('lookup'):
            self.source_strings = self.reverse_vocab_table.lookup(
                tf.cast(self.inputs.sources, tf.int64))
            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                self.target_strings = self.reverse_vocab_table.lookup(
                    tf.cast(self.inputs.targets_in, tf.int64))
                self.inference_strings = self.reverse_vocab_table.lookup(
                    tf.cast(self.inference_ids, tf.int64))
            else:
                self.inference_strings = self.reverse_vocab_table.lookup(
                    tf.cast(sentence_decoder_sample_ids, tf.int64))

        self.context_final_state = tf.reshape(
            context_encoder_state,
            tf.stack([self._batch_size, self._num_sentence, -1]))
        self.decoder_logit = sentence_decoder_logit
        self.decoder_sample_ids = sentence_decoder_sample_ids
        self.train_op = step
        self.learning_rate = lr

    def _build_sentence_encoder(self, inputs):
        sources = inputs.sources
        source_lengths = inputs.src_lengths
        batch_size = self._batch_size
        num_sentence = self._num_sentence

        flat_sources = tf.reshape(sources,
                                  tf.stack([batch_size * num_sentence, -1]))
        flat_lengths = tf.reshape(source_lengths, [-1])

        cell_type = self.hparams.rnn_cell_type
        num_rnn_layers = self.hparams.num_rnn_layers
        num_bidi_layers = self.hparams.num_bidi_layers
        num_uni_layers = num_rnn_layers - num_bidi_layers
        num_rnn_units = self.hparams.num_rnn_units
        dropout_keep_prob = self._dropout_keep_prob

        word_embedding = model_helper.create_word_embedding(
            num_vocab=self.hparams.num_vocab,
            embedding_dim=self.hparams.word_embedding_dim,
            name='encoder/word_embedding',
            pretrained_word_matrix=self.hparams.pretrained_word_path
        )
        rnn_inputs = tf.nn.embedding_lookup(word_embedding, flat_sources)
        rnn_lengths = flat_lengths

        assert (num_bidi_layers > 0 or num_uni_layers > 0)

        rnn_states = []
        # Current strategy: No residual at first layer
        if num_bidi_layers > 0:
            with tf.name_scope('bidi'):
                bidi_fw_cell, bidi_bw_cell = \
                    [model_helper.create_rnn_cell(
                        cell_type=cell_type,
                        num_layers=num_bidi_layers,
                        num_units=num_rnn_units // 2,
                        dropout_keep_prob=dropout_keep_prob,
                        num_residual_layers=num_bidi_layers - 1)
                     for _ in range(2)]
                (fw_outputs, bw_outputs), (fw_states, bw_states) = \
                    tf.nn.bidirectional_dynamic_rnn(
                        bidi_fw_cell, bidi_bw_cell,
                        rnn_inputs,
                        sequence_length=rnn_lengths,
                        dtype=tf.float32,
                        scope='sentence_encoder/bidirectional_rnn')

                rnn_outputs = rnn_inputs = tf.concat([fw_outputs, bw_outputs], 2)
                for fw_state, bw_state in zip(fw_states, bw_states):
                    if isinstance(fw_state, tf.contrib.rnn.LSTMStateTuple):
                        rnn_state = tf.contrib.rnn.LSTMStateTuple(
                            c=tf.concat([fw_state.c, bw_state.c], 1),
                            h=tf.concat([fw_state.h, bw_state.h], 1)
                        )
                    else:
                        rnn_state = tf.concat([fw_state, bw_state], 1)
                    rnn_states.append(rnn_state)

        if num_uni_layers > 0:
            with tf.name_scope('uni'):
                uni_cell = model_helper.create_rnn_cell(
                    cell_type=cell_type,
                    num_layers=num_uni_layers,
                    num_units=num_rnn_units,
                    dropout_keep_prob=dropout_keep_prob,
                    num_residual_layers=num_uni_layers-1)
                rnn_outputs, rnn_state = tf.nn.dynamic_rnn(
                    uni_cell,
                    rnn_inputs,
                    sequence_length=rnn_lengths,
                    dtype=tf.float32,
                    scope='sentence_encoder/unidirectional_rnn')
                rnn_states.append(rnn_state)

        # merge rnn states
        merged_rnn_state = []
        for rnn_state in rnn_states:
            if type(rnn_state) in [list, tuple]:
                merged_rnn_state.extend(list(rnn_state))
            else:
                merged_rnn_state.append(rnn_state)
        merged_rnn_state = tuple(merged_rnn_state)

        return rnn_outputs, merged_rnn_state

    def _build_context_encoder(self, sentence_encoder_states):
        batch_size = self._batch_size
        num_sentence = self._num_sentence

        state_inputs = sentence_encoder_states[-1]
        if isinstance(state_inputs, tf.contrib.rnn.LSTMStateTuple):
            state_inputs = tf.concat([state_inputs.c, state_inputs.h], 1)
            state_depth = self.hparams.num_rnn_units * 2
        else:
            state_depth = self.hparams.num_rnn_units

        state_inputs = tf.reshape(
            state_inputs,
            tf.stack([batch_size, num_sentence, state_depth]))
        context_cell = model_helper.create_rnn_cell(
            cell_type=self.hparams.rnn_cell_type,
            num_layers=1,
            num_units=self.hparams.num_rnn_units,
            dropout_keep_prob=self._dropout_keep_prob,
            num_residual_layers=0)

        batch_num_sentence = num_sentence * tf.ones(tf.stack([batch_size]),
                                                    dtype=tf.int32)
        context_outputs, context_state = tf.nn.dynamic_rnn(
            context_cell,
            state_inputs,
            sequence_length=batch_num_sentence,
            dtype=tf.float32,
            scope='context_encoder/unidirectional_rnn'
        )

        context_outputs = tf.reshape(context_outputs,
                                     tf.stack([batch_size * num_sentence, -1]))
        context_state = tf.reshape(context_state,
                                   tf.stack([batch_size * num_sentence, -1]))
        return context_outputs, context_state

    def _build_sentence_decoder(self, inputs,
                                context_encoder_outputs,
                                sentence_encoder_final_states,
                                sentence_encoder_outputs):
        batch_size = self._batch_size
        num_sentence = self._num_sentence

        word_embedding = model_helper.create_word_embedding(
            num_vocab=self.hparams.num_vocab,
            embedding_dim=self.hparams.word_embedding_dim,
            name='decoder/word_embedding',
            pretrained_word_matrix=self.hparams.pretrained_word_path
        )

        # tile_batch in inference mode
        beam_width = self.hparams.beam_width
        if self.mode == tf.contrib.learn.ModeKeys.INFER:
            # only decode last timestep
            if 'lstm' in self.hparams.rnn_cell_type.lower():
                batched_sentence_encoder_states = []
                for encoder_state in sentence_encoder_final_states:
                    target_shape = tf.stack([batch_size, num_sentence, -1])
                    c = s2s.tile_batch(
                        tf.reshape(encoder_state.c, target_shape)[:, -1, :],
                        beam_width)
                    h = s2s.tile_batch(
                        tf.reshape(encoder_state.h, target_shape)[:, -1, :],
                        beam_width)
                    batched_sentence_encoder_states.append(
                        tf.contrib.rnn.LSTMStateTuple(c=c, h=h))
            else:
                batched_sentence_encoder_states = [
                    s2s.tile_batch(
                        tf.reshape(
                            encoder_state,
                            tf.stack([batch_size, num_sentence, -1]))[:, -1, :],
                        beam_width)
                    for encoder_state in sentence_encoder_final_states
                ]
            sentence_encoder_final_states = tuple(batched_sentence_encoder_states)

            sentence_encoder_outputs = s2s.tile_batch(
                tf.reshape(
                    sentence_encoder_outputs,
                    tf.stack([batch_size,
                              num_sentence,
                              -1,
                              self.hparams.num_rnn_units]))[:, -1, :, :],
                beam_width)
            source_lengths = s2s.tile_batch(
                inputs.src_lengths[:, -1], beam_width)

            context_encoder_outputs = tf.reshape(
                context_encoder_outputs,
                tf.stack([batch_size,
                          num_sentence,
                          self.hparams.num_rnn_units]))[:, -1, :]
            context_encoder_outputs = tf.tile(
                tf.expand_dims(context_encoder_outputs, axis=1),
                [1, beam_width, 1]
            )
            effective_batch_size = self._batch_size * beam_width
        else:
            source_lengths = tf.reshape(inputs.src_lengths, [-1])
            context_encoder_outputs.set_shape([None, self.hparams.num_rnn_units])
            effective_batch_size = self._batch_size * self._num_sentence

        # Current strategy: No residual layers at decoder
        attention_mechanism = model_helper.create_attention_mechanism(
            attention_option=self.hparams.attention_type,
            num_units=self.hparams.num_rnn_units,
            memory=sentence_encoder_outputs,
            source_length=source_lengths)
        decoder_cell = s2s.AttentionWrapper(
            model_helper.create_rnn_cell(
                cell_type=self.hparams.rnn_cell_type,
                num_layers=self.hparams.num_rnn_layers,
                num_units=self.hparams.num_rnn_units,
                dropout_keep_prob=self._dropout_keep_prob,
                num_residual_layers=0),
            attention_mechanism,
            attention_layer_size=self.hparams.num_rnn_units,
            alignment_history=False,
            name="attention")

        decoder_initial_state = decoder_cell.zero_state(
            effective_batch_size, tf.float32)
        decoder_initial_state = decoder_initial_state.clone(
            cell_state=sentence_encoder_final_states)

        with tf.variable_scope('output_projection'):
            output_layer = layers_core.Dense(
                self.hparams.num_vocab, name="output_projection")
            self.output_layer = output_layer

        if self.mode in {tf.contrib.learn.ModeKeys.TRAIN,
                         tf.contrib.learn.ModeKeys.EVAL}:
            decoder_input_tokens = tf.reshape(
                inputs.targets_in, tf.stack([batch_size * num_sentence, -1]))
            decoder_inputs = tf.nn.embedding_lookup(
                word_embedding, decoder_input_tokens)
            target_lengths = tf.reshape(inputs.tgt_lengths, [-1])

            if self.mode == tf.contrib.learn.ModeKeys.TRAIN and False:
                sampling_probability = 1.0 - tf.train.exponential_decay(
                    1.0,
                    self.global_step,
                    self.hparams.scheduled_sampling_decay_steps,
                    self.hparams.scheduled_sampling_decay_rate,
                    staircase=True,
                    name='scheduled_sampling_prob'
                )
                helper = s2s.ScheduledEmbeddingTrainingHelper(
                    inputs=decoder_inputs,
                    sequence_length=target_lengths,
                    embedding=word_embedding,
                    sampling_probability=sampling_probability,
                    name='scheduled_sampling_helper'
                )
            else:
                helper = s2s.TrainingHelper(
                    inputs=decoder_inputs,
                    sequence_length=target_lengths,
                    name='training_helper',
                )
            decoder = s2s.BasicDecoder(
                decoder_cell, helper, decoder_initial_state, output_layer=None)
            final_outputs, final_state, _ = dynamic_decode_with_concat(
                decoder, context_encoder_outputs, swap_memory=True)
            logits = final_outputs.rnn_output
            sample_id = final_outputs.sample_id

        else:
            sos_id = tf.cast(self.vocab_table.lookup(tf.constant(dataset.SOS)),
                             tf.int32)
            eos_id = tf.cast(self.vocab_table.lookup(tf.constant(dataset.EOS)),
                             tf.int32)
            sos_ids = tf.fill([batch_size], sos_id)
            decoder = s2s.BeamSearchDecoder(
                cell=decoder_cell,
                embedding=word_embedding,
                start_tokens=sos_ids,
                end_token=eos_id,
                initial_state=decoder_initial_state,
                beam_width=beam_width,
                output_layer=self.output_layer
            )
            final_outputs, final_state, _ = dynamic_decode_with_concat(
                decoder, context_encoder_outputs,
                maximum_iterations=self.hparams.target_max_length,
                swap_memory=True)
            logits = final_outputs.beam_search_decoder_output.scores
            sample_id = final_outputs.predicted_ids

        return logits, final_state, sample_id

    def _build_loss(self, inputs, logits, class_weights, class_biases):
        target_lengths = tf.reshape(inputs.tgt_lengths, [-1])
        length_mask = tf.reshape(
            tf.sequence_mask(target_lengths, dtype=tf.float32), [-1])
        targets = tf.reshape(inputs.targets_out, [-1, 1])
        logits = tf.reshape(logits, [-1, self.hparams.num_rnn_units])
        flat_mask = tf.reshape(length_mask, [-1])

        sampler = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=tf.cast(targets, tf.int64),
            range_max=self.hparams.num_vocab,
            num_true=1,
            num_sampled=self.hparams.num_sample_softmax,
            unique=True,
            unigrams=(self.vocab_probs + 1e-16).tolist()
        )
        loss = tf.nn.sampled_softmax_loss(
            weights=class_weights,
            biases=class_biases,
            labels=targets,
            inputs=logits,
            num_sampled=self.hparams.num_sample_softmax,
            num_classes=self.hparams.num_vocab,
            num_true=1,
            sampled_values=sampler
        ) * flat_mask
        loss = tf.reduce_sum(loss) / (tf.reduce_sum(flat_mask) + 1e-12)

        #loss = s2s.sequence_loss(
        #    logits, targets, length_mask, name='sequence_loss')

        tf.summary.scalar('loss', loss)
        tf.summary.scalar('ppl', tf.exp(loss))

        return loss

    def _build_optimizer(self, loss):
        lr = tf.train.exponential_decay(
            self.hparams.initial_learning_rate,
            tf.maximum(0, self.global_step - self.hparams.start_decay_step),
            self.hparams.lr_decay_steps,
            self.hparams.lr_decay_rate,
            name='lr')
        lr = tf.maximum(
            lr,
            self.hparams.initial_learning_rate * self.hparams.min_decay_rate)

        if self.hparams.optimizer.lower() == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(lr)
        elif self.hparams.optimizer.lower() == 'momentum':
            optimizer = tf.train.MomentumOptimizer(lr,
                                                   self.hparams.momentum)
        elif self.hparams.optimizer.lower() == 'adam':
            optimizer = tf.train.AdamOptimizer(lr)
        else:
            raise ValueError('Unknown optimizer {}'.format(
                self.hparams.optimizer))

        vars = tf.trainable_variables()
        gradients = tf.gradients(loss, vars)
        gradients, global_norm = tf.clip_by_global_norm(
            gradients, self.hparams.max_grad_norm)
        step = optimizer.apply_gradients(
            zip(gradients, vars), global_step=self.global_step)

        tf.summary.scalar('learning_rate', lr)
        tf.summary.scalar('global_grad_norm', global_norm)

        return step, lr
