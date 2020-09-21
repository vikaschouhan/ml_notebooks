#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import random
from   modules.finance import *
from   modules.utils import *
import shutil


# In[ ]:


work_dir = 'out/direction_prediction_from_visual_patterns_notebook1__data'
shutil.rmtree(work_dir)
mkdir(work_dir)


# In[ ]:


# Get historical data
hist_data = load_pickle('data/data/nse_historical_25082020.pkl')


# In[ ]:


# Get keys for train & val
keys = list(hist_data.keys())
random.shuffle(keys)
train_keys = keys[:int(0.8*len(keys))]
val_keys   = keys[int(0.8*len(keys)):]

# Define train and val dir
train_dir  = os.path.join(work_dir, 'train')
val_dir    = os.path.join(work_dir, 'val')
train_annot_file = os.path.join(work_dir, 'train.csv')
val_annot_file   = os.path.join(work_dir, 'val.csv')


# In[ ]:


# Wrapper over generate_training_data_from_ticker
def gen_train_data(key, out_dir, num_samples_min=20, num_samples_max=400,
        forward_period=4, backward_period_min=40, backward_period_max=80,
        tick_name=None, timeout=20):
    # randomize number of samples
    num_samples    = random.randint(num_samples_min, num_samples_max)
    # randomize number of previous candles
    backward_period = random.randint(backward_period_min, backward_period_max)
    return generate_training_data_from_ticker(hist_data[key], out_dir=out_dir,
                                             num_samples=num_samples, forward_period=forward_period,
                                             backward_period=backward_period, tick_name=key,
                                             timeout=timeout, figscale=0.2)
# enddef


# In[ ]:


max_train_keys = 500
max_val_keys   = 200

# Generate multiple copies of train_keys selected at random
train_keys2 = []
for key_t in train_keys:
    train_keys2 += random.randint(1, 15) * [key_t]
# endfor
random.shuffle(train_keys2)

val_keys2 = []
for key_t in val_keys:
    val_keys2 += random.randint(1, 5) * [key_t]
# endfor
random.shuffle(val_keys2)

# Generate data with train_keys2
shutil.rmtree(train_dir, ignore_errors=True)
train_annots = spawn(gen_train_data, train_keys2[:max_train_keys], out_dir=train_dir)

shutil.rmtree(val_dir, ignore_errors=True)
val_annots = spawn(gen_train_data, val_keys2[:max_val_keys], out_dir=val_dir)
    
def generate_annot_file(annot, out_file):
    annot = {x:y for z in annot for x,y in z.items()}
    annot = pd.DataFrame(annot.items())
    annot.columns = ['image', 'return']
    annot = annot.set_index('image')
    annot.to_csv(out_file)
# enddef

# Generate annot files for train and val
generate_annot_file(train_annots, train_annot_file) 
generate_annot_file(val_annots, val_annot_file)


# In[ ]:


import tensorflow as tf
from   tensorflow.keras.layers import Dense, ReLU
from   tensorflow.keras import Model
import glob

####################################################
# Create fn for tf.data ingestor pipeline
####################################################
# Parse one file record
# record is a dictionary
def parse_one_record(record):
    # convert the compressed string to a 3D uint8 tensor
    record['image'] = tf.image.decode_image(tf.io.read_file(record['image']), channels=3, expand_animations=False)
    return record
# enddef

# Create dataset
# image data is dictionary
def _create_dataset(image_data):
    dh_dataset = tf.data.Dataset.from_tensor_slices(image_data)
    dh_dataset = dh_dataset.map(parse_one_record)
    return dh_dataset
# enddef


# Create data ingestor for tensorflow-2.x
def create_dataset(pkl_file, keys_list):
    # Load pickle file
    pkl_data  = load_pickle(pkl_file)
    # Only select keys which are actually present in pkl_file
    keys_list = list(set(keys_list) - set(pkl_data.keys()))
    
    # Convert image_data to format suitable for dataset ingestion
    image_data         = {
                             'image' : [os.path.join(image_dir, x) for x in metadata.index.to_list()],
                             'label' : [ 1 if x > 0 else 0 for x in metadata[metadata.columns[0]].to_list()],
                         }
    # Add few more info
    image_list         = image_data['image']
    image_id_list      = [os.path.basename(x) for x in image_list] # Get ids
    image_format_list  = [os.path.splitext(x)[1][1:] for x in image_id_list]
    num_samples        = len(image_list)

    image_data['imageformat']  = image_format_list
    image_data['imageid']      = image_id_list
    
    # Populate dataset and few extra info
    ret_value  = {}
    # Dataset
    ret_value['dataset']           = _create_dataset(image_data)
    ret_value['num_samples']       = num_samples
    return ret_value
# enddef


# In[ ]:


#####################################################
# Create fn for preprocessing pipeline.
#####################################################
def apply_image_normalization(image, normalize_type=0) :
    if normalize_type == 0: 
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0) # All pixels now between -1.0 and 1.0
        return image
    elif normalize_type == 1:
        image = tf.multiply(image, 2.0) # All pixels now between 0.0 and 2.0
        image = image - tf.reduce_mean(image, axis=[0, 1]) 
        # Most pixels should be between -1.0 and 1.0
        return image
    elif normalize_type == 2:
        image = tf.image.per_image_standardization(image)
        image = tf.multiply(image, 0.4) # This makes 98.8% of pixels between -1.0 and 1.0
        return image
    else :
        raise ValueError('invalid value for normalize_type: {}'.format(normalize_type))
    # endif
# enddef

def preprocess_image(image, size, normalize_type=2):
    # Convert float32
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, tf.float32)
    # endif
    
    # Resize to target size
    image     = tf.image.resize(image, [size, size], method='bilinear', antialias=False)
    # Apply normalization
    image     = apply_image_normalization(image, normalize_type)
    return image
# enddef

def create_preprocessing_function(image_size, normalize_type=2):
    def __wrap(record):
        record['image'] = preprocess_image(record['image'], image_size, normalize_type=normalize_type)
        return record
    # enddef

    return __wrap
# enddef


# In[ ]:


#####################################
# Create final data pipeline
####################################
tgt_image_size   = 64
train_batch_size = 32

# Generate pipeline
train_dataset = create_dataset(train_dir, train_annot_file)['dataset']
val_dataset   = create_dataset(val_dir, val_annot_file)['dataset']

# Apply preprocessing
prep_fn       = create_preprocessing_function(tgt_image_size)
train_dataset = train_dataset.map(prep_fn)
val_dataset   = val_dataset.map(prep_fn)

# Map from dict to tuples
train_dataset = train_dataset.map(lambda record: (record['image'], record['label']))
val_dataset   = val_dataset.map(lambda record: (record['image'], record['label']))

# Apply batch
train_dataset = train_dataset.batch(train_batch_size)
val_dataset   = val_dataset.batch(train_batch_size)


# In[ ]:


#######################################
# Create a tensorflow model. Initialize it and train it
##########################################
model_inp_shape = (tgt_image_size, tgt_image_size, 3)
model_t = tf.keras.applications.MobileNetV2(input_shape=model_inp_shape,
                                            classes=1,
                                            classifier_activation='softmax',
                                            include_top=False,
                                            pooling='max')


# In[ ]:


# Compile model
model_t.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])


# In[ ]:


# Run model
model_t.fit(train_dataset, validation_data=val_dataset, epochs=15)


# In[ ]:




