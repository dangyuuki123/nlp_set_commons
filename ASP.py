import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.backend import *
import keras

_BATCH_NORM_EPSILON = 1e-5
_BATCH_NORM_DECAY = 0.997
def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]

def merge_two_last_dims(x):
    b, d, f, c = shape_list(x)
    return tf.reshape(x, shape=[b, -1, f * c])
def ctc_lambda_func(args):

    y_pred, labels, input_length, label_length = args
    input_length = input_length //2
    return tf.reduce_mean(tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length))
vocabulary_size = 95
conv_type= "conv2d"
conv_kernels = [32 , 32 ,96  ]
conv_strides=[[2,2],[1,2],[1,2] ]
conv_filters=[[11,41] , [11,21] , [11,21]]
conv_dropout=0.5
rnn_nlayers= 5
nsubblocks =  2
padding = [[5,20],[5,10],[5,10]]
block_channels = [128,128,128,128,128]
block_kernels= [11, 13, 17, 21, 25]
block_dropout = 0.2
rnn_type= "lstm"
rnn_units= 1024
rnn_bidirectional=True
rnn_rowconv=  0
rnn_dropout= 0.0
fc_nlayers=  0
fc_units= 1600
fc_dropout=  0.1

class DeepSpeech(object):
    "ASP command"
    def __init__(self , num_rnn_layers = 3 , rnn_hidden_size = 1024 , num_classes = 95 , use_bias = True):
        self.num_rnn_layers = num_classes
        self.num_classes = num_classes
        self.rnn_hidden_size = rnn_hidden_size
        self.use_bias = use_bias
    def __call__(self , inputs , training):
        x = []
        # three cnn layers
        output = inputs
        for i in range(len(conv_kernels)): 
            output =tf.pad(
            output,
            [[0, 0], [padding[i][0], padding[i][0]], [padding[i][1], padding[i][1]], [0, 0]])  
            output = Conv2D(conv_kernels[i] , kernel_size= conv_filters[i] , strides =conv_strides[i]  , padding='same' , dilation_rate=1, dtype = tf.float32 )(output)
            output = tf.keras.layers.BatchNormalization(
            momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)(output)
            output = tf.keras.layers.LeakyReLU()(output)
            output = tf.keras.layers.Dropout(conv_dropout)(output)
        
        batch_size = tf.shape(output)[0]
        feat_size = output.get_shape().as_list()[2]
        tail = output.get_shape().as_list()[3]
        output = tf.reshape(
            output,
            [batch_size , -1 , feat_size * tail]
        )
        #Dense part 1 
        output = tf.keras.layers.Dense(fc_units)(output)
        output = tf.keras.layers.LeakyReLU()(output)
        x.append(output)
        
        # 2 conv1d forward
        for j in range(nsubblocks):
            for i in range(5):
                
                output = Conv1D(block_channels[i] , kernel_size= block_kernels[i] , strides =1  , padding='same' , dilation_rate=1, dtype = tf.float32)(output)
                output = tf.keras.layers.BatchNormalization()(output)
                output = tf.keras.layers.LeakyReLU()(output)
                output = tf.keras.layers.Dropout(block_dropout)(output)
            for k in range(len(x)):
                x[j] = Conv1D(block_channels[-1] , kernel_size= block_kernels[-1] , strides =1  , padding='same' , dilation_rate=1, dtype = tf.float32)(x[j])
                x[j] = tf.keras.layers.BatchNormalization()(x[j])
                output = tf.add(x[j] , output)
            x.append(output)
            output = tf.keras.layers.LeakyReLU()(output)
        # 3 LSTM 
        for i in range(self.num_rnn_layers):
            output = tf.keras.layers.BatchNormalization(
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)(output)
            lstm = tf.keras.layers.LSTM(rnn_units , dropout=rnn_dropout , return_sequences= True )
            output = tf.keras.layers.Bidirectional(lstm)(output)
        # full dense
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(fc_units))(output)
        output = tf.keras.layers.BatchNormalization(momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)(output)
        output = tf.keras.layers.LeakyReLU()(output)
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(fc_dropout))(output)
        logits = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=vocabulary_size, activation="softmax",
                                        use_bias=True ))(output)
        return logits


