'''Sequence to sequence example in Keras

English to Tagalog sentence pairs.
http://www.manythings.org/anki/tgl-eng.zip

Lots of neat sentence pairs datasets can be found at:
http://www.manythings.org/anki/

# References

- Sequence to Sequence Learning with Neural Networks
    https://arxiv.org/abs/1409.3215
- Learning Phrase Representations using
    RNN Encoder-Decoder for Statistical Machine Translation
    https://arxiv.org/abs/1406.1078
'''
from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from pyfasttext import FastText
import numpy as np


def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [content[i].split() for i in range(len(content))]
    content = np.array(content)
    content = np.reshape(content, [-1, ])
    return content

def build_dicts(words):
    dictionary = dict()
    for word in words:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

def build_seq2seq(latent_dim=256):
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None,))
    x = Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
    x, state_h, state_c = LSTM(latent_dim,
                               return_state=True)(x)
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None,))
    x = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
    x = LSTM(latent_dim, return_sequences=True)(x, initial_state=encoder_states)
    decoder_outputs = Dense(num_decoder_tokens, activation='softmax')(x)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

def build_models(latent_dim=256):
    return

def train_model():
    batch_size = 64  # Batch size for training.
    epochs = 1  # Number of epochs to train for.
    latent_dim = 256 # Latent dimensionality of the encoding space.
    # Path to the data txt file on disk.

    model = build_seq2seq()

    # Compile & run training
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    # Note that `decoder_target_data` needs to be one-hot encoded,
    # rather than sequences of integers like `decoder_input_data`!
    model.fit([encoder_input_data, decoder_input_data],
              decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2)

def input2target(data_path, sos, eos):
    input_texts = []
    target_texts = []

    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    for line in lines:
        if len(line) <= 0:
            continue
        #separate punctuation marks from words
        line = line.replace(",", " ,")
        line = line.replace(".", " .")
        line = line.replace("!", " !")
        line = line.replace("?", " ?")
        line = line.lower()
        target_text, input_text = line.split('\t')
        # print(input_text , " : ", target_text)
        #target_text = "%s %s %s" % (sos, target_text, eos)
        input_texts.append(input_text)
        target_texts.append(target_text)

    return input_texts, target_texts

def get_words(sentences):
    words = []
    for sen in sentences:
        tokens = sen.split()
        for token in tokens:
            if token not in words:
                words.append(token)
    print(len(words))
    return words

def sentence2tensor(input_texts, input_dict):
    return

def max_wordnum(texts):
    count = 0
    for text in texts:
        if len(text.split()) > count:
            count = len(text.split())
    return count


'''Load english and tagalog FastText embeddings'''
eng_fastText_path = '/media/raimarc/airscan4/datasets/fastText/wiki.en/wiki.en.bin'
fil_fastText_path = '/media/raimarc/airscan4/datasets/fastText/wiki.tl/wiki.tl.bin'
engModel = FastText(eng_fastText_path)
filModel = FastText(fil_fastText_path)

'''Parse Dataset'''
data_path = 'tgl-eng/tgl_ext.txt'
eos = "<EOS>"
sos = "<SOS>"
input_texts, target_texts = input2target(data_path, sos, eos)


input_words = get_words(input_texts)
input_dict, input_rev_dict = build_dicts(input_words)

target_words = get_words(target_texts)
if sos in target_words:
    print("Present")

target_dict, target_rev_dict = build_dicts(target_words)

embedding_dims = 300 # FastText has 300-dim vectors

num_encoder_tokens = len(input_words)
num_decoder_tokens = len(target_words)
max_encoder_seq_length = max([len(words.split()) for words in input_texts])
max_decoder_seq_length = max([len(words.split()) for words in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

encoder_input_data = np.zeros(
            (len(input_texts), max_encoder_seq_length, embedding_dims),
                dtype='float32')
print("encshape",encoder_input_data.shape)
decoder_input_data = np.zeros(
            (len(input_texts), max_decoder_seq_length, embedding_dims),
                dtype='float32')

print("decshape",decoder_input_data.shape)
decoder_target_data = np.zeros(
            (len(input_texts), max_decoder_seq_length, embedding_dims),
                dtype='float32')

for i, text, in enumerate(input_texts):
    words = text.split()
    for t, word in enumerate(words):
        encoder_input_data[i, t, :] = filModel.get_numpy_vector(word)

for i, text, in enumerate(target_texts):
    words = text.split()
    for t, word in enumerate(words):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, :] = engModel.get_numpy_vector(word)

        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t-1, :] = engModel.get_numpy_vector(word)

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 512 # Latent dimensionality of the encoding space.

# Path to the data txt file on disk.
# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, embedding_dims))
encoder = LSTM(latent_dim, return_sequences=True)(encoder_inputs)
encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(encoder)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, embedding_dims))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(embedding_dims, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()

# Compile & run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
# Note that `decoder_target_data` needs to be one-hot encoded,
# rather than sequences of integers like `decoder_input_data`!

model.fit([encoder_input_data, decoder_input_data],
          decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1)

# Save model
#model.save('s2s.h5')
model.save('s2s-moredata.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)
encoder_model.summary()

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)
decoder_model.summary()

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, embedding_dims))
    # Populate the first character of target sequence with the start character.
    #target_seq[0, 0, :] = engModel.get_numpy_vector(sos)

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    ctr = 0
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        #print(output_tokens.shape)

        # Sample a token
        #sampled_token_index = np.argmax(output_tokens[0, -1, :])
        #sampled_word = target_rev_dict[sampled_token_index]
        sampled_word = engModel.words_for_vector(output_tokens[0,0,:])
        #print("sampled word", sampled_word)
        decoded_sentence += sampled_word[0][0] + " "

        # Exit condition: either hit max length
        # or find stop character.
        # if sampled_word in [".", "?", "!"] or

        ctr += 1
        if (sampled_word == eos or
           len(decoded_sentence) > max_decoder_seq_length or
           ctr >= max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, embedding_dims))
        #target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

indexes = np.random.randint(0, len(input_texts), 100)

for seq_index in indexes:
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
