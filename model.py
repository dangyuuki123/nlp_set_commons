import tensorflow as tf
from keras.callbacks import *
from keras.layers import *
from keras.models import *
def cliped_relu():
    return tf.keras.activations.relu(x , max_value = 20 )
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    print(y_pred.shape)
    print(labels.shape)
    print(input_length.shape)
    print(label_length.shape)
    return tf.keras.backend.ctc_batch_cost(labels , y_pred , input_length , label_length)
def ctc(y_true , y_pred):
    return y_pred
class SpeechModel(object):


    def __init__(self, hparams):
        rnn_size = 512
        input_tensor = Input(name='inputs', shape=[hparams['max_input_length'], 221, 1])
        x = input_tensor
        x_shape = x.get_shape()
        print(x.get_shape())
        x = Convolution2D(filters=32, kernel_size=(11, 3), strides=1, activation='relu', padding='same')(x)
        # print(x.get_shape())
        conv_shape = x.get_shape()
        x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)
        # print(x.get_shape())

        gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)
        gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True,
                     kernel_initializer='he_normal', name='gru1_b')(x)
        x = concatenate([gru_1, gru_1b])

        gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(x)
        gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True,
                     kernel_initializer='he_normal', name='gru2_b')(x)
        x = concatenate([gru_2, gru_2b])

        gru_3 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru3')(x)
        gru_3b = GRU(rnn_size, return_sequences=True, go_backwards=True,
                     kernel_initializer='he_normal', name='gru3_b')(x)
        x = concatenate([gru_3, gru_3b])

        gru_4 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru4')(x)
        gru_4b = GRU(rnn_size, return_sequences=True, go_backwards=True,
                     kernel_initializer='he_normal', name='gru4_b')(x)
        x = concatenate([gru_4, gru_4b])
        
        x = BatchNormalization()(x)
        
        x = Dense(1024, activation='relu')(x)
        # x = Dropout(0.05)(x)
        x = Dense(95, activation='softmax', name='base_model_out')(x)
        #base_model = Model(inputs=input_tensor, outputs=x)
        #plot_model(base_model, to_file="model.png", show_shapes=True)
        # base_model.summary()
        # return
        labels = Input(name='labels', shape=hparams['max_label_length'], dtype='int32')
        input_length = Input(name='input_lengths', shape=[1], dtype='int64')
        label_length = Input(name='label_lengths', shape=[1], dtype='int64')
        loss_out = Lambda(ctc_lambda_func, output_shape=(1,),
                          name='ctc')([x, labels, input_length, label_length])
        self.model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=[loss_out])
        self.model.summary()

        optimizer = tf.keras.optimizers.Adam(lr=hparams['learning_rate'], beta_1=0.9, beta_2=0.999,
                                                 epsilon=1e-8, clipnorm=5)

        self.model.compile(optimizer=optimizer, loss={'ctc': lambda y_true, y_pred: y_pred} )

    def train_generator(self, generator, train_params):

        callbacks = []

        if train_params['tensorboard']:
            callbacks.append(tf.keras.callbacks.TensorBoard(train_params['log_dir'], write_images=True))

        self.model.fit(generator, epochs=train_params['epochs'],
                                 steps_per_epoch=train_params['steps_per_epoch'])