# Setting up a tf.data pipeline
# To ensure that the model is fed with data efficiently we will be using tf.data API to create our input pipeline. The input pipeline consists for the following major processing steps:

# Apply the preprocessing function to the samples
# Create batches with fixed batch size. Since images in the batch can have different dimensions, and can also have different number of objects, we use padded_batch to the add the necessary padding to create rectangular tensors
# Create targets for each sample in the batch using LabelEncoder

import numpy as np
import tensorflow as tf
from tensorflow import keras

from preprocess import preprocess_data
# from encode import LabelEncoder 


autotune = tf.data.experimental.AUTOTUNE

def train_data_loader(train_dataset, label_encoder, batch_size) :
    train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=autotune)
#     train_dataset = train_dataset.shuffle(8 * batch_size)
    train_dataset = train_dataset.shuffle(100)
    train_dataset = train_dataset.padded_batch(
        batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
    )
    train_dataset = train_dataset.map(
        label_encoder.encode_batch, num_parallel_calls=autotune
    )
    train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
#     train_dataset = train_dataset.prefetch(autotune)
    return train_dataset

def val_data_loader(val_dataset, label_encoder, batch_size=1) :
    val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
    val_dataset = val_dataset.padded_batch(
        batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True
    )
    val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
    val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
    val_dataset = val_dataset.prefetch(autotune)
    
    return val_dataset

