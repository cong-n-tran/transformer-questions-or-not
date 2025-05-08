# Khanh Nguyen Cong Tran
# 1002046419

import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np


def train_transformer(train_inputs, train_labels, validation_inputs, validation_labels):

    # vocab format
    vocab_size      = 250
    sequence_length = 8     

    # professor's hyperparameters
    embed_dim       = 100
    num_heads       = 3
    ff_dim          = 32
    epochs          = 10
    batch_size      = 32


    # create the text vectorization
    text_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length
    )
    text_vectorization.adapt(train_inputs)

    # vectorized the input
    x_train = text_vectorization(train_inputs)
    x_val   = text_vectorization(validation_inputs)

    # create the input
    inputs      = layers.Input(shape=(sequence_length,), dtype='int32')

    # create embedding layer and connect from input (input -> embedd)
    token_embed = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)

    # create the positional embedding
    positions            = tf.range(start=0, limit=sequence_length, delta=1)
    position_embedding   = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)
    pos_embed            = position_embedding(positions)  # shape (sequence_length, embed_dim)
    
    # conconate postional layer and embedding layer
    x = token_embed + pos_embed  

    # create the attention head
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    attn_output = layers.Dropout(0.1)(attn_output)
    
    # add layer normalization after the attention head + x
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

    # add a layer so the network can learn after the transformer layers
    ffn = tf.keras.Sequential([
        layers.Dense(ff_dim, activation='relu'),
        layers.Dense(embed_dim),
    ])

    # more dropout and normalization
    ffn_output = ffn(x)
    ffn_output = layers.Dropout(0.1)(ffn_output)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

    # pooling 1d 
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    # test if the input to output is correct
    model = keras.Model(inputs=inputs, outputs=outputs)

    # compile 
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy', # because our output is binary (true or false)
        metrics=['accuracy']
    )
    # train
    model.fit(
        x_train, train_labels,
        validation_data=(x_val, validation_labels),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2
    )

    return model, text_vectorization


def evaluate_transformer(model, text_vectorization, test_inputs, test_labels):
    # vectorized the test_inputs
    x_test = text_vectorization(test_inputs)
    
    # evaluating using the test_labels
    loss, accuracy = model.evaluate(x_test, test_labels, verbose=0)

    # return accuracy
    return accuracy
