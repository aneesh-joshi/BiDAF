from keras.layers import Input, Embedding, Dense, Concatenate, TimeDistributed, LSTM, Bidirectional, Lambda, Reshape, Masking
from keras.models import Model
import keras.backend as K
from custom_layers import Highway, MatrixAttention, MaskedSoftmax, WeightedSum, RepeatLike, Max, ComplexConcat
import numpy as np

num_question_words = 4
num_passage_words = 5
max_passage_sents = 7
num_highway_layers = 2
highway_activation = 'relu'
num_hidden_ending_bidir_layers = 2
embedding_dim = 100
n_encoder_hidden_nodes = 42

indexed_question = np.array([[1, 3, 69, 69], [3, 5, 2, 69]])
indexed_passage = np.array([[[2, 69, 69, 69, 69],
                             [12, 45, 23, 6, 69],
                             [69, 69, 69, 69, 69],
                             [12, 45, 23, 6, 69],
                             [69, 69, 69, 69, 69],
                             [12, 45, 23, 6, 69],
                             [69, 69, 69, 69, 69]],
                             [[2, 69, 69, 69, 69],
                             [12, 45, 23, 6, 69],
                             [69, 69, 69, 69, 69],
                             [12, 45, 23, 6, 69],
                             [69, 69, 69, 69, 69],
                             [12, 45, 23, 6, 69],
                             [69, 69, 69, 69, 69]]])
y = np.array([[[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
                [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]])

max_passage_words = num_passage_words*max_passage_sents

indexed_passage = indexed_passage.reshape(2, max_passage_words)

print(indexed_passage.shape)
print(indexed_question.shape)
print(y.shape)



question_input = Input(shape=(num_question_words,), dtype='int32', name="question_input")
passage_input = Input(shape=(max_passage_words,), dtype='int32', name="passage_input")

embedding_layer = Embedding(input_dim=70, output_dim=embedding_dim)



question_embedding = embedding_layer(question_input)  # (None, num_question_words, embedding_dim)
passage_embedding = embedding_layer(passage_input)  # (None, num_passage_words*max_passage_sents, embedding_dim)

# masking_layer = Masking(mask_value=69)
# question_embedding = masking_layer(question_embedding)
# passage_embedding = masking_layer(passage_embedding)

for i in range(num_highway_layers):
    highway_layer = Highway(activation=highway_activation, name='highway_{}'.format(i))
    
    question_layer = TimeDistributed(highway_layer, name=highway_layer.name + "_qtd")
    question_embedding = question_layer(question_embedding)

    passage_layer = TimeDistributed(highway_layer, name=highway_layer.name + "_ptd")
    passage_embedding = passage_layer(passage_embedding)


passage_bidir_encoder = TimeDistributed(Bidirectional(LSTM(n_encoder_hidden_nodes, return_sequences=True,
                                                           name='PassageBidirEncoder'), merge_mode='concat'))
# question_bidir_encoder = Bidirectional(LSTM(n_encoder_hidden_nodes, return_sequences=True, name='QuestionBidirEncoder'), merge_mode='concat')

encoded_passage = passage_bidir_encoder(passage_embedding)
encoded_question = passage_bidir_encoder(question_embedding)


# PART 2:
# Now we compute a similarity between the passage words and the question words, and
# normalize the matrix in a couple of different ways for input into some more layers.
matrix_attention_layer = MatrixAttention(name='passage_question_similarity')
# Shape: (batch_size, num_passage_words, num_question_words)
passage_question_similarity = matrix_attention_layer([encoded_passage, encoded_question])

# Shape: (batch_size, num_passage_words, num_question_words), normalized over question
# words for each passage word.
passage_question_attention = MaskedSoftmax()(passage_question_similarity)
# Shape: (batch_size, num_passage_words, embedding_dim * 2)
weighted_sum_layer = WeightedSum(name="passage_question_vectors", use_masking=False)
passage_question_vectors = weighted_sum_layer([encoded_question, passage_question_attention])

# Min's paper finds, for each document word, the most similar question word to it, and
# computes a single attention over the whole document using these max similarities.
# Shape: (batch_size, num_passage_words)
question_passage_similarity = Max(axis=-1)(passage_question_similarity)
# Shape: (batch_size, num_passage_words)
question_passage_attention = MaskedSoftmax()(question_passage_similarity)
# Shape: (batch_size, embedding_dim * 2)
weighted_sum_layer = WeightedSum(name="question_passage_vector", use_masking=False)
question_passage_vector = weighted_sum_layer([encoded_passage, question_passage_attention])

# Then he repeats this question/passage vector for every word in the passage, and uses it
# as an additional input to the hidden layers above.
repeat_layer = RepeatLike(axis=1, copy_from_axis=1)
# Shape: (batch_size, num_passage_words, embedding_dim * 2)
tiled_question_passage_vector = repeat_layer([question_passage_vector, encoded_passage])

# Shape: (batch_size, num_passage_words, embedding_dim * 8)
complex_concat_layer = ComplexConcat(combination='1,2,1*2,1*3', name='final_merged_passage')
final_merged_passage = complex_concat_layer([encoded_passage,
                                             passage_question_vectors,
                                             tiled_question_passage_vector])

# PART 3:
# Having computed a combined representation of the document that includes attended question
# vectors, we'll pass this through a few more bi-directional encoder layers, then predict
# the span_begin word.  Hard to find a good name for this; Min calls this part of the
# network the "modeling layer", so we'll call this the `modeled_passage`.
modeled_passage = final_merged_passage
for i in range(num_hidden_ending_bidir_layers):
    hidden_layer = Bidirectional(LSTM(n_encoder_hidden_nodes, return_sequences=True, name='EndingBiDirEncoder_{}'.format(i)), merge_mode='concat')
    modeled_passage = hidden_layer(modeled_passage)

# To predict the span word, we pass the merged representation through a Dense layer without
# output size 1 (basically a dot product of a vector of weights and the passage vectors),
# then do a softmax to get a position.
span_begin_input = Concatenate()([final_merged_passage, modeled_passage])
span_begin_input = Reshape((max_passage_sents, num_passage_words, -1))(span_begin_input)
maxxed = Max(axis=-1)(span_begin_input)
prediction = Dense(2, activation='softmax')(maxxed)

model = Model(inputs=[question_input, passage_input], outputs=[prediction])
model.summary()
# print(model.predict(x={'question_input': indexed_question, 'passage_input': indexed_passage}))

# model.compile(optimizer='adam', loss='categorical_crossentropy')
# model.fit(x={'question_input': indexed_question, 'passage_input': indexed_passage}, y=y)

# print(model.predict(x={'question_input': indexed_question, 'passage_input': indexed_passage}))

# # Shape: (batch_size, num_passage_words)
# span_begin_probabilities = MaskedSoftmax(name="span_begin_softmax")(span_begin_weights)


# sum_layer = WeightedSum(name="passage_weighted_by_predicted_span", use_masking=False)
# repeat_layer = RepeatLike(axis=1, copy_from_axis=1)
# passage_weighted_by_predicted_span = repeat_layer([sum_layer([modeled_passage, span_begin_probabilities]),
#                                                    encoded_passage])

# span_end_representation = ComplexConcat(combination="1,2,3,2*3")([final_merged_passage,
#                                                                   modeled_passage,
#                                                                   passage_weighted_by_predicted_span])


# final_seq2seq = Bidirectional(LSTM(embedding_dim, return_sequences=True, name="final_seq2seq"), merge_mode='concat')
# span_end_representation = final_seq2seq(span_end_representation)
# span_end_input = Concatenate()([final_merged_passage, span_end_representation])
# span_end_weights = TimeDistributed(Dense(units=1))(span_end_input)
# span_end_probabilities = MaskedSoftmax(name="span_end_softmax")(span_end_weights)

# # encoded_passage = Lambda(lambda x: K.tile(K.expand_dims(x, 1), [1, num_question_words, 1, 1]), name="passage_repeat")(encoded_passage)
# # encoded_question = Lambda(lambda x: K.tile(K.expand_dims(x, 1), [1, num_passage_words, 1, 1]), name="question_repeat")(encoded_question)

# model = Model(inputs=[question_input, passage_input], outputs=[span_begin_probabilities, span_end_probabilities])
# model.summary()
# print(model.predict([indexed_question, indexed_passage]))
# # Shape: (batch_size, num_question_words, embedding_dim * 2) (embedding_dim * 2 because,
# # by default in this class, _embed_input concatenates a word embedding with a
# # character-level encoder).
# question_embedding = _embed_input(question_input)

# # Shape: (batch_size, num_passage_words, embedding_dim * 2)
# passage_embedding = _embed_input(passage_input)