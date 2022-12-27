import os
import pdb
import random

import numpy as np
import tensorflow as tf


def savePasswords(passwords, num_generate, temperature):

    file_name = 'passwords/passwords_'+str(num_generate) + '_'+str(temperature)+'.txt'

    try:
        passwd = open(file_name, 'w')
        for i in passwords:
            passwd.write(i)

    except:
        print('Erro open file')

    passwd.close


def generate_text(model, num_generate, temperature, start_string):

    input_indices = [char2index[s] for s in start_string]
    input_indices = tf.expand_dims(input_indices, 0)

    text_generated = []

    model.reset_states()
    for char_index in range(num_generate):
        predictions = model(input_indices)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # Using a categorical distribution to predict the character returned by the model.
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(
            predictions,
            num_samples=1
        )[-1, 0].numpy()

        input_indices = tf.expand_dims([predicted_id], 0)

        text_generated.append(index2char[predicted_id])

    savePasswords(text_generated, num_generate, temperature)


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        batch_input_shape=[batch_size, None]
    ))

    # Mudar de acordo com o modelo

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


def getSeeds(lines, number_of_seeds):

    seeds = []

    for i in range(number_of_seeds):
        seeds.append(random.choice(lines))

    return '\n'.join(seeds)


text = open('train.txt', mode='r', encoding="latin-1").read()
print('Length of text: {} characters'.format(len(text)))


test = open('test.txt', mode='r', encoding="latin-1").read()
lines = test.splitlines()

seeds = getSeeds(lines, 15)

print("Seeds: ", seeds)

vocab = sorted(set(text))
vocab_size = len(vocab)

char2index = {char: index for index, char in enumerate(vocab)}
index2char = np.array(vocab)
print(index2char)


embedding_dim = 256

# Number of RNN units.
rnn_units = 1024

checkpoint_dir = 'tmp/checkpoints'

tf.train.latest_checkpoint(checkpoint_dir)

simplified_batch_size = 1

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)


model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([simplified_batch_size, None]))

model.summary()


num_generate = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
temperature = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
               0.7, 0.8, 0.7, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]


for num in num_generate:
    for temp in temperature:
        print('Number Generate: ', str(num) + '\n' +
              'Temperature: ' + str(temp) + '\n')
        print(seeds)
        generate_text(model, num, temp, start_string=seeds)
