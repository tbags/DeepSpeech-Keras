from typing import List
from keras import Model
import tensorflow as tf
from tensorflow import set_random_seed
from keras.initializers import np
from keras.backend.tensorflow_backend import expand_dims, squeeze
from keras.layers import Input, Lambda, LSTM, CuDNNLSTM, Bidirectional, Dense, ReLU, TimeDistributed, \
                         BatchNormalization, Dropout, ZeroPadding2D, Conv2D, Reshape


def deepspeech(is_gpu: bool, input_dim=80, output_dim=36, context=7,
               units=1024, dropouts=[0.1, 0.1, 0], random_state=1) -> Model:
    """
    Model is adapted from: Deep Speech: Scaling up end-to-end speech recognition (https://arxiv.org/abs/1412.5567)

    It is worth to know which default parameters are used:
    - Conv2D:
        strides: (1, 1)
        padding: "valid"
        dilatation_rate: 1
        activation: "linear"
        use_bias: True
        data_format: "channels last"
        kernel_initializer: "glorot_uniform"
        bias_initializer: "zeros"
    - Dense:
        activation: "linear"
        use_bias: True
        kernel_initializer: "glorot_uniform"
        bias_initializer: "zeros"
    - LSTM (as for CuDNNLSTM):
        use_bias: True,
        kernel_initializer: "glorot_uniform"
        recurrent_initializer: "orthogonal"
        bias_initializer: "zeros"
        unit_forget_bias: True
        implementation: 1
        return_state: False
        go_backwards: False
        stateful: False
        unroll: False
    """
    np.random.seed(random_state)
    set_random_seed(random_state)                                                   # Create model under CPU scope and avoid OOM
    with tf.device('/cpu:0'):                                                       # erors during concatenation a large distributed model.
        input_tensor = Input([None, input_dim], name='X')                           # Define input tensor [time, features]
        x = Lambda(expand_dims, arguments=dict(axis=-1))(input_tensor)              # Add 4th dim (channel)
        x = ZeroPadding2D(padding=(context, 0))(x)                                  # Fill zeros around time dimension
        receptive_field = (2*context + 1, input_dim)                                # Take into account fore/back-ward context
        x = Conv2D(filters=units, kernel_size=receptive_field)(x)                   # Convolve signal in time dim
        x = Lambda(squeeze, arguments=dict(axis=2))(x)                              # Squeeze into 3rd dim array
        x = ReLU(max_value=20)(x)                                                   # Add non-linearity
        x = Dropout(rate=dropouts[0])(x)                                            # Use dropout as regularization

        x = TimeDistributed(Dense(units))(x)                                        # 2nd and 3rd FC layers do a feature
        x = ReLU(max_value=20)(x)                                                   # extraction base on the context
        x = Dropout(rate=dropouts[1])(x)

        x = TimeDistributed(Dense(units))(x)
        x = ReLU(max_value=20)(x)
        x = Dropout(rate=dropouts[2])(x)

        x = Bidirectional(CuDNNLSTM(units, return_sequences=True) if is_gpu else     # LSTM handle long dependencies
                          LSTM(units, return_sequences=True, ),
                          merge_mode='sum')(x)

        output_tensor = TimeDistributed(Dense(output_dim, activation='softmax'))(x)  # Return at each time step prob along characters
        model = Model(input_tensor, output_tensor, name='DeepSpeech')
    return model


def deepspeech_custom(is_gpu: bool, layers: List[dict], input_dim: int, to_freeze: List[dict] = [], random_state=1) -> Model:
    np.random.seed(random_state)
    set_random_seed(random_state)

    constructors = {
        'BatchNormalization': lambda params: BatchNormalization(**params),
        'Conv2D': lambda params: Conv2D(**params, name=name),
        'Dense': lambda params: TimeDistributed(Dense(**params), name=name),
        'Dropout': lambda params: Dropout(**params),
        'LSTM': lambda params: Bidirectional(CuDNNLSTM(**params) if is_gpu else
                                             LSTM(activation='tanh', recurrent_activation='sigmoid', **params),
                                             merge_mode='sum', name=name),
        'ReLU': lambda params: ReLU(**params),
        'ZeroPadding2D': lambda params: ZeroPadding2D(**params),
        'expand_dims': lambda params: Lambda(expand_dims, arguments=params),
        'squeeze': lambda params: Lambda(squeeze, arguments=params),
        'squeeze_last_dims': lambda params: Reshape([-1, params['units']])
    }
    with tf.device('/cpu:0'):
        input_tensor = Input([None, input_dim], name='X')

        x = input_tensor
        for params in layers:
            constructor_name = params.pop('constructor')
            name = params.pop('name') if 'name' in params else None     # `name` is implicit passed to constructors
            constructor = constructors[constructor_name]                # Conv2D, TimeDistributed and Bidirectional.
            layer = constructor(params)
            x = layer(x)
        output_tensor = x

        model = Model(input_tensor, output_tensor, name='DeepSpeech')
        for params in to_freeze:
            name = params.pop('name')
            layer = model.get_layer(name)
            layer.trainable = False
    return model

