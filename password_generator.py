import os
import pathlib
import pdb
import platform
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Activation, Dense, Dropout
from sklearn.model_selection import train_test_split


def generate_text(model, start_string, num_generate = 1000, temperature=1.0):

    input_indices = [char2index[s] for s in start_string]
    input_indices = tf.expand_dims(input_indices, 0)

    text_generated = []

    model.reset_states()

    for char_index in range(num_generate):
        predictions = model(input_indices)
        predictions = tf.squeeze(predictions, 0)

        predictions = predictions / temperature
        predicted_id = tf.random.categorical(
        predictions,
        num_samples=1
        )[-1,0].numpy()

        input_indices = tf.expand_dims([predicted_id], 0)

        text_generated.append(index2char[predicted_id])

    return (start_string + ''.join(text_generated))


def render_training_history(training_history):
    loss = training_history.history['loss']
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(loss, label='Training set')
    plt.legend()
    plt.grid(linestyle='--', linewidth=1, alpha=0.5)
    plt.show()

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(
      y_true=labels,
      y_pred=logits,
      from_logits=True
    )

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Embedding(
      input_dim=vocab_size,
      output_dim=embedding_dim,
      batch_input_shape=[batch_size, None]
    ))

    model.add(tf.keras.layers.LSTM(
      units=rnn_units,
      return_sequences=True,
      stateful=True,
      recurrent_initializer=tf.keras.initializers.GlorotNormal()
    ))

    model.add(tf.keras.layers.LSTM(
       units=rnn_units,
       return_sequences=True,
       stateful=True,
       recurrent_initializer=tf.keras.initializers.GlorotNormal()
    ))


    model.add(tf.keras.layers.Dense(vocab_size))

    return model

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


text = open('train.txt', mode='r', encoding="latin-1").read()
print('Length of text: {} characters'.format(len(text)))

vocab = sorted(set(text))

print('{} unique characters'.format(len(vocab)))
print('vocab:', vocab)

char2index = {char: index for index, char in enumerate(vocab)}
index2char = np.array(vocab)


text_as_int = np.array([char2index[char] for char in text])

sequence_length=25
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(sequence_length + 1, drop_remainder=True)
dataset = sequences.map(split_input_target)

BATCH_SIZE = 32

BUFFER_SIZE = 1000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print('Batched dataset size: {}'.format(len(list(dataset.as_numpy_iterator()))))

vocab_size = len(vocab)

embedding_dim = 256

rnn_units = 512

model = build_model(vocab_size, embedding_dim, rnn_units, BATCH_SIZE)
model.summary()

tf.keras.utils.plot_model(
    model,
    show_shapes=True,
    show_layer_names=True,
)


adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)
model.compile(optimizer=adam_optimizer, loss=loss,  metrics=['accuracy'])

checkpoint_dir = 'tmp/checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True
)

EPOCHS=1000

history = model.fit(
  x=dataset,
  epochs=EPOCHS,
  callbacks=[
    checkpoint_callback,
    EarlyStopping(monitor='loss', verbose=1, mode='min', patience=5)
  ]
)

render_training_history(history)

tf.train.latest_checkpoint(checkpoint_dir)

simplified_batch_size = 1

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([simplified_batch_size, None]))

model.summary()

