import os 
import csv 
import numpy as np
import tensorflow as tf
import numpy as np
import re
import re
def re_clean(text):
    #c.append('')
    #if len(c)%1000==0:
        #pass
        #print(len(c))
    text=text.lower()
    symbol = """!#$%^&*();:\t\\\"!\{\}\[\]<>-\?\-\\\"—\.,1234567890"""
    text=re.sub('\n','', text)
    #text=' '.join([stem.stem(i) for i in text.split() if not i in stop])
    #text=[i for i in text.split() if i not in stop]
    return text #' '.join(text)
def create_character_mapping():
    character_map = {' ': 0}
    f = open('vocabulary.txt')
    a = f.readlines()
    for i in a:
        i = re_clean(i)
        character_map[i] = len(character_map)

    return character_map
def get_data_details(filename):
    result = {
        'max_input_length': 0,
        'max_label_length': 0,
        'num_samples': 0
    }

    # Get max lengths
    with open(filename, 'r') as metadata:
        metadata_reader = csv.DictReader(metadata, fieldnames=['filename', 'spec_length', 'labels_length', 'labels'])
        next(metadata_reader)
        for row in metadata_reader:
            if int(row['spec_length']) > result['max_input_length']:
                result['max_input_length'] = int(row['spec_length'])
            if int(row['labels_length']) > result['max_label_length']:
                result['max_label_length'] = int(row['labels_length'])
            result['num_samples'] += 1
    return result 
def create_data_generator(directory, max_input_length, max_label_length, batch_size=16 , filename):
    x, y, input_lengths, label_lengths = [], [], [], [] 
    with (open(os.path.join(directory, filename), 'r')) as metadata:
        metadata_reader = csv.DictReader(metadata, fieldnames=['filename', 'spec_length', 'labels_length', 'labels'])
        next(metadata_reader)
        for row in metadata_reader:
            if(os.path.exists(os.path.join(directory  , row['filename'] +'.npy'))!=1):
                continue
            audio = np.load(os.path.join(directory  , row['filename'] +'.npy'))
            x.append(audio)
            #x = np.array(x)
            m = row['labels'].split(' ')
            k = []
            for i in m :
                i = int(i)
                k.append(i)
            y.append(k)
            #y = np.array(y)
            input_lengths.append(int(row['spec_length']))
            label_lengths.append(int(row['labels_length']))
            if len(x) == batch_size:
                yield {
                    'inputs': tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_input_length, padding='post' , dtype = 'float32'),
                    'labels': tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=max_label_length, padding='post'),
                    'input_lengths': np.asarray(input_lengths),
                    'label_lengths': np.asarray(label_lengths)
                }, {
                    'ctc': np.zeros([batch_size])
                }
                x, y, input_lengths, label_lengths = [], [], [], []
