from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq.python.ops.decoder import Decoder


_transpose_batch_time = rnn._transpose_batch_time


def _create_zero_outputs(size, dtype, batch_size):
    """Create a zero outputs Tensor structure."""
    def _t(s):
        return (s if isinstance(s, ops.Tensor) else constant_op.constant(
            tensor_shape.TensorShape(s).as_list(),
            dtype=dtypes.int32,
            name="zero_suffix_shape"))

    def _create(s, d):
        return array_ops.zeros(
            array_ops.concat(
                ([batch_size], _t(s)), axis=0), dtype=d)

    return nest.map_structure(_create, size, dtype)


def dynamic_decode_with_concat(decoder,
                               to_concat,
                               output_time_major=False,
                               impute_finished=False,
                               maximum_iterations=None,
                               parallel_iterations=32,
                               swap_memory=False,
                               scope=None):
    """Perform dynamic decoding with `decoder`,
    with additional concat of `to_concat` in inputs.
    Base code: dynamic_decode from tf 1.3"""

    if not isinstance(decoder, Decoder):
        raise TypeError("Expected decoder to be type Decoder, but saw: %s" %
                        type(decoder))

    with variable_scope.variable_scope(scope, "decoder") as varscope:
        # Properly cache variable values inside the while_loop
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)

        if maximum_iterations is not None:
            maximum_iterations = ops.convert_to_tensor(
                maximum_iterations,
                dtype=dtypes.int32,
                name="maximum_iterations")
            if maximum_iterations.get_shape().ndims != 0:
                raise ValueError("maximum_iterations must be a scalar")

        initial_finished, initial_inputs, initial_state = decoder.initialize()
        initial_inputs = array_ops.concat([initial_inputs, to_concat], axis=-1)
        zero_outputs = _create_zero_outputs(decoder.output_size,
                                            decoder.output_dtype,
                                            decoder.batch_size)

        if maximum_iterations is not None:
            initial_finished = math_ops.logical_or(
                initial_finished, 0 >= maximum_iterations)
        initial_sequence_lengths = array_ops.zeros_like(
            initial_finished, dtype=dtypes.int32)
        initial_time = constant_op.constant(0, dtype=dtypes.int32)

        def _shape(batch_size, from_shape):
            if not isinstance(from_shape, tensor_shape.TensorShape):
                return tensor_shape.TensorShape(None)
            else:
                batch_size = tensor_util.constant_value(
                    ops.convert_to_tensor(
                        batch_size, name="batch_size"))
                return tensor_shape.TensorShape(
                    [batch_size]).concatenate(from_shape)

        def _create_ta(s, d):
            return tensor_array_ops.TensorArray(
                dtype=d,
                size=0,
                dynamic_size=True,
                element_shape=_shape(decoder.batch_size, s))

        initial_outputs_ta = nest.map_structure(_create_ta, decoder.output_size,
                                                decoder.output_dtype)

        def condition(unused_time, unused_outputs_ta, unused_state, unused_inputs,
                      finished, unused_sequence_lengths):
            return math_ops.logical_not(math_ops.reduce_all(finished))

        def body(time, outputs_ta, state, inputs, finished, sequence_lengths):
            (next_outputs, decoder_state, next_inputs,
             decoder_finished) = decoder.step(time, inputs, state)
            next_finished = math_ops.logical_or(decoder_finished, finished)
            if maximum_iterations is not None:
                next_finished = math_ops.logical_or(
                    next_finished, time + 1 >= maximum_iterations)
            next_sequence_lengths = array_ops.where(
                math_ops.logical_and(math_ops.logical_not(finished), next_finished),
                array_ops.fill(array_ops.shape(sequence_lengths), time + 1),
                sequence_lengths)

            nest.assert_same_structure(state, decoder_state)
            nest.assert_same_structure(outputs_ta, next_outputs)
            nest.assert_same_structure(inputs, next_inputs)

            # Zero out output values past finish
            if impute_finished:
                emit = nest.map_structure(
                    lambda out, zero: array_ops.where(finished, zero, out),
                    next_outputs,
                    zero_outputs)
            else:
                emit = next_outputs

            # Copy through states past finish
            def _maybe_copy_state(new, cur):
                # TensorArrays and scalar states get passed through.
                if isinstance(cur, tensor_array_ops.TensorArray):
                    pass_through = True
                else:
                    new.set_shape(cur.shape)
                    pass_through = (new.shape.ndims == 0)
                return new if pass_through else array_ops.where(finished, cur, new)

            if impute_finished:
                next_state = nest.map_structure(
                    _maybe_copy_state, decoder_state, state)
            else:
                next_state = decoder_state

            outputs_ta = nest.map_structure(lambda ta, out: ta.write(time, out),
                                            outputs_ta, emit)
            next_inputs = array_ops.concat([next_inputs, to_concat], axis=-1)

            return (time + 1, outputs_ta, next_state, next_inputs, next_finished,
                    next_sequence_lengths)

        res = control_flow_ops.while_loop(
            condition,
            body,
            loop_vars=[
                initial_time, initial_outputs_ta, initial_state, initial_inputs,
                initial_finished, initial_sequence_lengths,
            ],
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory)

        final_outputs_ta = res[1]
        final_state = res[2]
        final_sequence_lengths = res[5]

        final_outputs = nest.map_structure(lambda ta: ta.stack(),
                                           final_outputs_ta)

        try:
            final_outputs, final_state = decoder.finalize(
                final_outputs, final_state, final_sequence_lengths)
        except NotImplementedError:
            pass

        if not output_time_major:
            final_outputs = nest.map_structure(_transpose_batch_time,
                                               final_outputs)

    return final_outputs, final_state, final_sequence_lengths
