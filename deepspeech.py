import tensorflow as tf
from tensorflow.keras.layers import*
from tensorflow.keras.models import*
from tensorflow.keras.backend import *
from sequence_wise_bn import SequenceBatchNorm
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
    with tf.GradientTape() as tape:
        tape.watch(y_pred)
    return tf.reduce_mean(tf.keras.backend.ctc_batch_cost(labels , y_pred , input_length , label_length))
def SpeechModel (model,
                 name: str = "deepspeech2"):
        #super(ConvModule, self).__init__(**kwargs)

    vocabulary_size = 95
    conv_type= "conv2d"
    conv_kernels = [256, 384, 512, 640, 768]
    conv_strides=[11, 13, 17, 21, 25]
    conv_filters=[11, 13, 17, 21, 25]
    conv_dropout=[0.2, 0.2, 0.2, 0.3, 0.3]
    rnn_nlayers= 5
    rnn_type= "lstm"
    rnn_units= 128
    rnn_bidirectional=True
    rnn_rowconv=  0
    rnn_dropout= 0.1
    fc_nlayers=  0
    fc_units= 1024
    fc_dropout=  0.1
    assert len(conv_kernels) == len(conv_strides) == len(conv_filters)
    #assert dropout >= 0.0 
    input_ = tf.keras.Input(name = 'inputs' , shape = (model['max_input_length'] , 80))
    output = Conv1D(256 , kernel_size= 11 , strides = 2 , padding='same' , dilation_rate=1, dtype = tf.float32)(input_)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.ReLU()(output)
    output = tf.keras.layers.Dropout(0.2)(output)
    x = output 
    for j in range(3):        
        for i in range(len(conv_kernels)):   
            output = SeparableConv1D(conv_kernels[i] , kernel_size= conv_fillters[i] , strides = 1 , padding='same' , dilation_rate=1, dtype = tf.float32)(input_)
            output = tf.keras.layers.BatchNormalization()(output)
            output = tf.keras.layers.ReLU()(output)
            output = tf.keras.layers.Dropout(conv_dropout[i])(output) 
         output =  SeparableConv1D(conv_kernels[-1] , kernel_size= conv_fillters[-1] , strides = 1 , padding='same' , dilation_rate=1, dtype = tf.float32)(input_)
         output =  BatchNormalization()(output)
         x = Conv1D(conv_kernels[-1] , kernel_size= conv_fillters[-1] , strides = 1 , padding='same' , dilation_rate=1, dtype = tf.float32)(x)
         x = BatchNormalization()(x)
         output = tf.keras.layers.add([x , output])
         output = tf.keras.layers.ReLU()(output)
         output = tf.keras.layers.Dropout(0.2)(output) 
         x = output
         
    for i in range(3):
        lstm = tf.keras.layers.LSTM(rnn_units , dropout = rnn_dropout ,  return_sequences=True , use_bias=True)
        output = tf.keras.layers.Bidirectional(lstm )(output)
        output = SequenceBatchNorm(time_major=False)(output)
        output = tf.keras.layers.Dropout(fc_dropout)(output)
    output = Conv1D(896 , kernel_size= 1 , strides = 1 , padding='same' , dilation_rate=2, dtype = tf.float32)(input_)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.ReLU()(output)
    output = tf.keras.layers.Dropout(0.4)(output)
    output = tf.keras.layers.Dense(fc_units)(output)
    
    output = Conv1D(1024 , kernel_size= 1 , strides = 1 , padding='same' , dilation_rate=1, dtype = tf.float32)(input_)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.ReLU()(output)
    output = tf.keras.layers.Dropout(0.4)(output)
    
    output = tf.keras.layers.Dense(fc_units)(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.ReLU()(output)
    output = tf.keras.layers.Dropout(fc_dropout)(output)
    output = tf.keras.layers.Dense(units=vocabulary_size, activation="softmax",
                                    use_bias=True)(output)
    labels = Input(name='labels', shape=model['max_label_length'], dtype='int32')
    input_length = Input(name='input_lengths', shape=[1], dtype='int64')
    label_length = Input(name='label_lengths', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,),
                      name='ctc')([output, labels, input_length, label_length])
    return tf.keras.Model(inputs=[input_, labels, input_length, label_length], outputs=[loss_out]) , tf.keras.Model(inputs=input_ , outputs=output)
            
        
            
