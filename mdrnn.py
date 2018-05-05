import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple
from mdlstm import MDLSTMCell


def multi_dimensional_rnn_while_loop(rnn_size, input_data, sh, dims=None, scope_n="layer1"):
    """Implements naive multi dimension recurrent neural networks

    @param rnn_size: the hidden units
    @param input_data: the data to process of shape [batch,h,w,channels]
    @param sh: [height,width] of the windows
    @param dims: dimensions to reverse the input data,eg.
        dims=[False,True,True,False] => true means reverse dimension
    @param scope_n : the scope

    returns [batch,h/sh[0],w/sh[1],rnn_size] the output of the lstm
    """

    with tf.variable_scope("MultiDimensionalLSTMCell-" + scope_n):

        # Create multidimensional cell with selected size
        cell = MDLSTMCell(rnn_size)

        # Get the shape of the imput (batch_size, x, y, channels)
        shape = input_data.get_shape().as_list()
        batch_size = shape[0]
        X_dim = shape[1]
        Y_dim = shape[2]
        channels = shape[3]
        # Window size
        X_win = sh[0]
        Y_win = sh[1]
        # Get the runtime batch size
        batch_size_runtime = tf.shape(input_data)[0]

        # If the imput cannot be exactly sampled by the window, we patch it
        # with zeros
        if X_dim % X_win != 0:
            # Get offset size
            offset = tf.zeros([batch_size_runtime, X_win -
                               (X_dim % X_win), Y_dim, channels])
            # Concatenate X dimension
            input_data = tf.concat(axis=1, values=[input_data, offset])
            # Get new shape
            shape = input_data.get_shape().as_list()
            # Update shape value
            X_dim = shape[1]

        # The same but for Y axis
        if Y_dim % Y_win != 0:
            # Get offset size
            offset = tf.zeros([batch_size_runtime, X_dim,
                               Y_win - (Y_dim % Y_win), channels])
            # Concatenate Y dimension
            input_data = tf.concat(axis=2, values=[input_data, offset])
            # Get new shape
            shape = input_data.get_shape().as_list()
            # Update shape value
            Y_dim = shape[2]

        # Get the steps to perform in X and Y axis
        h, w = int(X_dim / X_win), int(Y_dim / Y_win)

        # Get the number of features (total number of imput values per step)
        features = Y_win * X_win * channels

        # Reshape input data to a tensor containing the step indexes and features inputs
        # The batch size is inferred from the tensor size
        x = tf.reshape(input_data, [batch_size_runtime, h, w, features])

        # Reverse the selected dimensions
        if dims is not None:
            assert dims[0] is False and dims[3] is False
            x = tf.reverse(x, dims)

        # Reorder inputs to (h, w, batch_size, features)
        x = tf.transpose(x, [1, 2, 0, 3])
        # Reshape to a one dimensional tensor of (h*w*batch_size , features)
        x = tf.reshape(x, [-1, features])
        # Split tensor into h*w tensors of size (batch_size , features)
        x = tf.split(axis=0, num_or_size_splits=h * w, value=x)

        # Create an input tensor array (literally an array of tensors) to use
        # inside the loop
        inputs_ta = tf.TensorArray(
            dtype=tf.float32, size=h * w, name='input_ta')
        # Unestack the input X in the tensor array
        inputs_ta = inputs_ta.unstack(x)
        # Create an input tensor array for the states
        states_ta = tf.TensorArray(
            dtype=tf.float32, size=h * w + 1, name='state_ta', clear_after_read=False)
        # And an other for the output
        outputs_ta = tf.TensorArray(
            dtype=tf.float32, size=h * w, name='output_ta')

        # initial cell hidden states
        # Write to the last position of the array, the LSTMStateTuple filled
        # with zeros
        states_ta = states_ta.write(h * w, LSTMStateTuple(tf.zeros([batch_size_runtime, rnn_size], tf.float32),
                                                          tf.zeros([batch_size_runtime, rnn_size], tf.float32)))

        # Function to get the sample skipping one row
        def get_up(t_, w_):
            return t_ - tf.constant(w_)
        # Function to get the previous sample

        def get_last(t_, w_):
            return t_ - tf.constant(1)

        # Controls the initial index
        time = tf.constant(0)
        zero = tf.constant(0)

        # Body of the while loop operation that aplies the MD LSTM
        def body(time_, outputs_ta_, states_ta_):

            # If the current position is less or equal than the width, we are in the first row
            # and we need to read the zero state we added in row (h*w).
            # If not, get the sample located at a width dstance.
            state_up = tf.cond(tf.less_equal(time_, tf.constant(w)),
                               lambda: states_ta_.read(h * w),
                               lambda: states_ta_.read(get_up(time_, w)))

            # If it is the first step we read the zero state if not we read the
            # inmediate last
            state_last = tf.cond(tf.less(zero, tf.mod(time_, tf.constant(w))),
                                 lambda: states_ta_.read(get_last(time_, w)),
                                 lambda: states_ta_.read(h * w))

            # We build the input state in both dimensions
            current_state = state_up[0], state_last[
                0], state_up[1], state_last[1]
            # Now we calculate the output state and the cell output
            out, state = cell(inputs_ta.read(time_), current_state)
            # We write the output to the output tensor array
            outputs_ta_ = outputs_ta_.write(time_, out)
            # And save the output state to the state tensor array
            states_ta_ = states_ta_.write(time_, state)

            # Return outputs and incremented time step
            return time_ + 1, outputs_ta_, states_ta_

        # Loop output condition. The index, given by the time, should be less than the
        # total number of steps defined within the image
        def condition(time_, outputs_ta_, states_ta_):
            return tf.less(time_, tf.constant(h * w))

        # Run the looped operation
        result, outputs_ta, states_ta = tf.while_loop(condition, body, [time, outputs_ta, states_ta],
                                                      parallel_iterations=1)

        # Extract the output tensors from the processesd tensor array
        outputs = outputs_ta.stack()
        states = states_ta.stack()

        # Reshape outputs to match the shape of the imput
        y = tf.reshape(outputs, [h, w, batch_size_runtime, rnn_size])

        # Reorder te dimensions to match the input
        y = tf.transpose(y, [2, 0, 1, 3])
        # Reverse if selected
        if dims is not None:
            y = tf.reverse(y, dims)

        # Return the output and the inner states
        return y, states
