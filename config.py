preprocessing = {

    'data_dir': 'spec/',
    'window_size': 20,
    'step_size': 10,

}

model = {

    'verbose': 1,

    'conv_channels': [32 , 32 , 64 , 128],
    'conv_filters': [13 , 11, 7 , 5 ],
    'conv_strides': [1,1,1,1],

    'rnn_units': [1024 , 1024,1024,1024 , 1024 ,1024],
    'bidirectional_rnn': True,

    'future_context': 2,

    'use_bn': True,

    'learning_rate': 0.001

}

training = {

    'tensorboard': False,
    'log_dir': './logs',

    'batch_size': 8,
    'epochs': 300,
    'validation_size': 0.2

}
