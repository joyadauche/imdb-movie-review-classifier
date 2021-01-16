import tensorflow as tf
import os
import numpy as np
import shutil
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Activation, Dense, Embedding, GlobalAveragePooling1D

vocab_size = 10000
embedding_dim = 64
max_length = 16
trunc_type = 'post'
oov_tok = "<OOV>"

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url, untar=True)
dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
os.listdir(dataset_dir)

train_dir = os.path.join(dataset_dir, 'train')

remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

batch_size = 1024
seed = 123
train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    train_dir, batch_size=batch_size, validation_split=0.2,
    subset='training', seed=seed)
val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    train_dir, batch_size=batch_size, validation_split=0.2,
    subset='validation', seed=seed)


AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

training_sentences = []
training_labels = []
testing_sentences = []
testing_labels = []
for text_batch, label_batch in train_ds:
    for i in range(len(text_batch)):
        training_sentences.append(str(text_batch.numpy()[i]))
        training_labels.append(label_batch[i].numpy())
training_labels_final = np.array(training_labels)

for text_batch, label_batch in val_ds:
    for i in range(len(text_batch)):
        testing_sentences.append(str(text_batch.numpy()[i]))
        testing_labels.append(label_batch[i].numpy())
testing_labels_final = np.array(testing_labels)

print(type(training_sentences))
print(training_sentences[0:2])

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length)

print('padded: ', padded[0])
print('test padded: ', testing_padded[0])

model = Sequential([
  Embedding(vocab_size, embedding_dim, name="embedding"),
  GlobalAveragePooling1D(),
  Dense(16, activation='relu'),
  Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()
model.fit(
    padded,
    training_labels_final,
    validation_data=(testing_padded, testing_labels_final),
    epochs=50)
model.save("imdb-movie-review-classifier.h5")