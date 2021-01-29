from keras import backend as K
from keras.engine.base_layer import Layer, InputSpec
from keras import activations
from keras.layers.convolutional import _Conv
import numpy as np

class TDNN(_Conv):
    # Original TDNN
    # A. Waibel, T. Hanazawa, G. Hinton, K. Shikano and K. J. Lang,
    # "Phoneme recognition using time-delay neural networks,"
    # IEEE Transactions on Acoustics, Speech, and Signal Processing,
    # vol. 37, no. 3, pp. 328-339, March 1989.
    
    # Architecture implemented in this Layer
    # Peddinti, Vijayaditya et al.
    # "A time delay neural network architecture
    # for efficient modeling of long temporal contexts."
    # INTERSPEECH (2015).

    def __init__(self,
                 input_context,
                 sub_sampling=False,
                 filters=1,
                 strides=1,
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if padding == 'causal':
            if data_format != 'channels_last':
                raise ValueError('When using causal padding in `Conv1D`, '
                                 '`data_format` must be "channels_last" '
                                 '(temporal data).')
        self.input_context = input_context
        self.sub_sampling = sub_sampling
        super(TDNN, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=(self.input_context[1]-self.input_context[0]+1,),
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
    
    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        if self.sub_sampling:
            self.mask = np.zeros(kernel_shape)
            self.mask[0][0] = 1
            self.mask[self.input_context[1]-self.input_context[0]][0] = 1
        else:
            self.mask = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        if self.sub_sampling:
            self.kernel *= self.mask
        return super(TDNN, self).call(inputs)


    def get_config(self):
        config = super(TDNN, self).get_config()
        config.pop('rank')
        return config
