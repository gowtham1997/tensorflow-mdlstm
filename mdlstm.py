import tensorflow as tf
from tensorflow.contrib.rnn import LayerRNNCell, LSTMStateTuple
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.layers import base as base_layer

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

# Adapted from https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/ops/rnn_cell_impl.py
# and
# https://github.com/philipperemy/tensorflow-multi-dimensional-lstm/blob/master/md_lstm.py


class MDLSTMCell(LayerRNNCell):
    """
    Multi Dimensional LSTM recurrent network cell.
    The implementation is based on: https://arxiv.org/pdf/0705.2011.pdf.
    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.
    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.

    Please be aware that state_is_tuple is always true.
    """

    def __init__(self, num_units, forget_bias=1.0, activation=None, reuse=None, name=None, dimensions=2):
        """Initialize the multi dimensional LSTM cell.
        Args:
          num_units: int, The number of units in the LSTM cell.
          forget_bias: float, The bias added to forget gates (see above).
            Must set to `0.0` manually when restoring from CudnnLSTM-trained
            checkpoints.
          activation: Activation function of the inner states.  Default: `tanh`.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.
          name: String, the name of the layer. Layers with the same name will
            share weights, but to avoid mistakes we require reuse=True in such
            cases.
          When restoring from CudnnLSTM-trained checkpoints, must use
          `CudnnCompatibleLSTMCell` instead.
        """
        super(MDLSTMCell, self).__init__(_reuse=reuse, name=name)
        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)

        self._num_units = num_units
        self._dimensions = dimensions
        self._forget_bias = forget_bias
        self._activation = activation or math_ops.tanh

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value
        h_depth = self._num_units
        self._kernel = self.add_variable(
            _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + h_depth * self._dimensions, (3 + self._dimensions) * self._num_units])
        self._bias = self.add_variable(
            _BIAS_VARIABLE_NAME,
            shape=[(3 + self._dimensions) * self._num_units],
            initializer=init_ops.zeros_initializer(dtype=self.dtype))

        self.built = True

    def call(self, inputs, state):
        """Long short-term memory cell (LSTM).
        Args:
          inputs: `2-D` tensor with shape `[batch_size, input_size]`.
          state: An `LSTMStateTuple` of state tensors, each shaped
            `[batch_size, self.state_size]`
        Returns:
          A pair containing the new hidden state, and the new state (`LSTMStateTuple`).
        """
        sigmoid = math_ops.sigmoid
        one = constant_op.constant(1, dtype=tf.int32)
        # Parameters of gates are concatenated into one multiply for
        # efficiency.
        c_slice = state[0:self._dimensions]
        h_slice = state[self._dimensions:self._dimensions * 2]
        c1, c2, h1, h2 = state

        concat_ar = [inputs]
        concat_ar.extend(h_slice)
        gate_inputs = math_ops.matmul(
            array_ops.concat(concat_ar, 1), self._kernel)  # last should be replaced by _kernel
        gate_inputs = nn_ops.bias_add(gate_inputs, self._bias)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        gate_array = array_ops.split(
            value=gate_inputs, num_or_size_splits=(3 + self._dimensions), axis=one)
        i, j = gate_array[0:2]
        f = gate_array[2:2 + self._dimensions]
        o = gate_array[2 + self._dimensions]

        forget_bias_tensor = constant_op.constant(
            self._forget_bias, dtype=f[0].dtype)
        # Note that using `add` and `multiply` instead of `+` and `*` gives a
        # performance improvement. So using those at the cost of readability.
        add = math_ops.add
        add_n = math_ops.add_n
        multiply = math_ops.multiply
        new_c = add_n([multiply(c_slice[idx], sigmoid(
            add(f[idx], forget_bias_tensor))) for idx in range(len(f))])
        new_c = add(new_c,
                    multiply(sigmoid(i), self._activation(j)))
        new_h = multiply(self._activation(new_c), sigmoid(o))

        new_state = LSTMStateTuple(new_c, new_h)
        return new_h, new_state
