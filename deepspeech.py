import tensorflow as tf
from tensorflow.keras.layers import*
from tensorflow.keras.models import*
from tensorflow.keras.backend import *
from sequence_wise_bn import SequenceBatchNorm
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

def SpeechModel (model,
                 name: str = "deepspeech2"):
        #super(ConvModule, self).__init__(**kwargs)

    vocabulary_size = 28
    conv_type= "conv2d"
    conv_kernels = [32 , 32 ,96  ]
    conv_strides=[[2,2],[1,2],[1,2] ]
    conv_filters=[[11,41] , [11,21] , [11,21]]
    conv_dropout=0.5
    rnn_nlayers= 5
    nsubblocks =  2
    padding = [[5,20] , [5,10] ,[5,10]]
    block_channels = [256, 384, 512, 640, 768]
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
    assert len(conv_kernels) == len(conv_strides) == len(conv_filters)
    x = []
    #assert dropout >= 0.0 
    input_ = tf.keras.Input(name = 'inputs' , shape = (model['max_input_length'] , 221 , 1 ))
    output = input_
    
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
    
    output = keras.layers.Masking()(output)
    
    x = output
    output = tf.keras.layers.Dense(fc_units)(output)
    output = tf.keras.layers.LeakyReLU()(output)
    output = tf.keras.layers.Dense(fc_units)(output)
    x = tf.keras.layers.Dense(fc_units)(x)
    output = keras.layers.add([output ,x])
    output = tf.keras.layers.Dense(fc_units)(output)
    output = tf.keras.layers.Dense(fc_units)(output)
    output = tf.keras.layers.LeakyReLU()(output)
    #output = tf.keras.layers.ZeroPadding1D(padding=(0, 1711))(output)
   
    for i in range(5):
        output = tf.keras.layers.BatchNormalization(
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)(output)
        lstm = tf.keras.layers.LSTM(rnn_units , dropout = rnn_dropout ,  return_sequences=True , use_bias=True)
        output = tf.keras.layers.Bidirectional(lstm )(output)

        
        
    output = tf.keras.layers.Dense(fc_units)(output)
    output = tf.keras.layers.BatchNormalization(
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)(output)
    output = tf.keras.layers.LeakyReLU()(output)
    output = tf.keras.layers.Dropout(fc_dropout)(output)
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=vocabulary_size, activation="softmax",
                                    use_bias=True ))(output)
    labels = Input(name='labels', shape=model['max_label_length'], dtype='int64')
    input_length = Input(name='input_lengths', shape=[1], dtype='int64')
    label_length = Input(name='label_lengths', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,),
                      name='ctc')([output, labels, input_length, label_length])
    return tf.keras.Model(inputs=[input_, labels, input_length, label_length], outputs=[loss_out]) , tf.keras.Model(inputs=input_ , outputs=output)
            
        
        
            
