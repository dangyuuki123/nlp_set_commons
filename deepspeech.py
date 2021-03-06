import tensorflow as tf
from tensorflow.keras.layers import*
from tensorflow.keras.models import*
from tensorflow.keras.backend import *
from sequence_wise_bn import SequenceBatchNorm
import keras

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

    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)

def SpeechModel (model,
                 name: str = "deepspeech2"):
        #super(ConvModule, self).__init__(**kwargs)

    vocabulary_size = 95
    conv_type= "conv2d"
    conv_kernels = [32 , 32 ,96  ]
    conv_strides=[[2,2],[1,2],[1,2] ]
    conv_filters=[[41 ,11] , [21 ,11] , [11,11]]
    conv_dropout=0.1
    rnn_nlayers= 5
    nsubblocks =  8
    block_channels = [256, 384, 512, 640, 768]
    block_kernels= [11, 13, 17, 21, 25]
    block_dropout = 0.1
    rnn_type= "lstm"
    rnn_units= 1024
    rnn_bidirectional=True
    rnn_rowconv=  0
    rnn_dropout= 0.1
    fc_nlayers=  0
    fc_units= 1024
    fc_dropout=  0.1
    assert len(conv_kernels) == len(conv_strides) == len(conv_filters)
    x = []
    #assert dropout >= 0.0 
    input_ = tf.keras.Input(name = 'inputs' , shape = (model['max_input_length'] , 80 , 1 ))
    output = input_
    
    for i in range(len(conv_kernels)):   
        output = Conv2D(conv_kernels[i] , kernel_size= conv_filters[i] , strides =conv_strides[i]  , padding='same' , dilation_rate=1, dtype = tf.float32)(output)
        output = tf.keras.layers.BatchNormalization()(output)
        output = tf.keras.layers.LeakyReLU()(output)
        output = tf.keras.layers.Dropout(conv_dropout)(output)
        
    output = merge_two_last_dims(output)
    output = tf.keras.layers.Dense(1024  ,activation='sigmoid')(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dense(1024  ,activation='sigmoid')(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.Dense(1024  ,activation='sigmoid')(output)
    output = tf.keras.layers.BatchNormalization()(output)
    x.append(output)
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
        output = tf.keras.layers.Dropout(0.1)(output)
    for i in range(8):
        lstm = tf.keras.layers.LSTM(rnn_units , dropout = rnn_dropout ,  return_sequences=True , use_bias=True)
        output = tf.keras.layers.Bidirectional(lstm )(output)
        output = SequenceBatchNorm(time_major=False)(output)
        output = tf.keras.layers.Dropout(fc_dropout)(output)
        
    output = tf.keras.layers.Dense(fc_units)(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.LeakyReLU()(output)
    output = tf.keras.layers.Dropout(fc_dropout)(output)
    output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=vocabulary_size, activation="softmax",
                                    use_bias=True))(output)
    labels = Input(name='labels', shape=model['max_label_length'], dtype='int64')
    input_length = Input(name='input_lengths', shape=[1], dtype='int64')
    label_length = Input(name='label_lengths', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,),
                      name='ctc')([output, labels, input_length, label_length])
    return tf.keras.Model(inputs=[input_, labels, input_length, label_length], outputs=[loss_out]) , tf.keras.Model(inputs=input_ , outputs=output)
            
        
        
            
