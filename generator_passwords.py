import pdb

import numpy as np
import tensorflow as tf


def generate_text(model, start_string, num_generate = 400000, temperature=1.0):
    # Evaluation step (generating text using the learned model)
    # Converting our start string to numbers (vectorizing).
    input_indices = [char2index[s] for s in start_string]
    input_indices = tf.expand_dims(input_indices, 0)

    # Empty string to store our results.
    text_generated = []

    # Here batch size == 1.
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
        )[-1,0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state.
        input_indices = tf.expand_dims([predicted_id], 0)

        text_generated.append(index2char[predicted_id])

    return (start_string + ''.join(text_generated))


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

    # model.add(tf.keras.layers.LSTM(
    #   units=rnn_units,
    #   return_sequences=True,
    #   stateful=True,
    #   recurrent_initializer=tf.keras.initializers.GlorotNormal()
    # ))


    model.add(tf.keras.layers.Dense(vocab_size))


    return model




text = open('train.txt', mode='r', encoding="latin-1").read()
print('Length of text: {} characters'.format(len(text)))

# The unique characters in the file
vocab = sorted(set(text))
vocab_size = len(vocab)

char2index = {char: index for index, char in enumerate(vocab)}
index2char = np.array(vocab)
print(index2char)

# The embedding dimension.
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

print(generate_text(model, start_string="mastermaster\njanuary7\nunobedient\npardoned\ntyrosinase\nsottishness\nrichierich\nbaronessa\njoaninha\nemprosthotonos"))
#print(generate_text(model, start_string="password"))
