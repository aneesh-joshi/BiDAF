import logging
import numpy as np
import hashlib
from numpy import random as np_random
from gensim.models import KeyedVectors
from collections import Counter
from custom_losses import rank_hinge_loss
import tensorflow as tf
from evaluation_metrics import mapk, mean_ndcg
from sklearn.preprocessing import normalize
from gensim import utils
from collections import Iterable

from keras.layers import Input, Embedding, Dense, Concatenate, TimeDistributed, LSTM, Bidirectional, Lambda, Activation
from keras.models import Model
import keras.backend as K
from custom_layers import Highway, MatrixAttention, MaskedSoftmax, WeightedSum, RepeatLike, Max, ComplexConcat


try:
    import keras.backend as K
    from keras import optimizers
    from keras.models import load_model
    from keras.losses import hinge
    from keras.models import Model
    from keras.layers import Input, Embedding, Dot, Dense, Reshape, Dropout
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False

logger = logging.getLogger(__name__)




def _get_full_batch_iter(pair_list, batch_size):
    """Provides all the data points int the format: X1, X2, y with
    alternate positive and negative examples of `batch_size` in a streamable format.

    Parameters
    ----------
    pair_list : iterable list of tuple
                See docstring for _get_pair_list for more details
    batch_size : int
        half the size in which the generator will yield datapoints. The size is doubled since
        we include positive and negative examples.

    Yields
    -------
    X1 : numpy array of shape (batch_size * 2, text_maxlen)
        the queries
    X2 : numpy array of shape (batch_size * 2, text_maxlen)
        the docs
    y : numpy array with {0, 1} of shape (batch_size * 2, 1)
        The relation between X1[i] and X2[j]
        1 : X2[i] is relevant to X1[i]
        0 : X2[i] is not relevant to X1[i]
    """

    X1, X2, y = [], [], []
    while True:
        j=0
        for i, (query, pos_doc, neg_doc) in enumerate(pair_list):
            X1.append(query)
            X2.append(pos_doc)
            y.append(1)
            X1.append(query)
            X2.append(neg_doc)
            y.append(0)
            j+=1
            if i % batch_size == 0 and i != 0:
                yield ({'query': np.array(X1), 'doc': np.array(X2)}, np.array(y))
                X1, X2, y = [], [], []


def _get_pair_list(queries, docs, labels, _make_indexed, is_iterable):
    """Yields a tuple with query document pairs in the format
    (query, positive_doc, negative_doc)
    [(q1, d+, d-), (q2, d+, d-), (q3, d+, d-), ..., (qn, d+, d-)]
        where each query or document is a list of int

    Parameters
    ----------
    queries : iterable list of list of str
        The queries to the model
    docs : iterable list of list of list of str
        The candidate documents for each query
    labels : iterable list of int
        The relevance of the document to the query. 1 = relevant, 0 = not relevant
    _make_indexed : function
        Translates the given sentence as a list of list of str into a list of list of int
        based on the model's internal dictionary
    is_iterable : bool
        Whether the input data is streamable

    Example
    -------
    [(['When', 'was', 'Abraham', 'Lincoln', 'born', '?'],
      ['He', 'was', 'born', 'in', '1809'],
      ['Abraham', 'Lincoln', 'was', 'the', 'president',
       'of', 'the', 'United', 'States', 'of', 'America']),

     (['When', 'was', 'the', 'first', 'World', 'War', '?'],
      ['It', 'was', 'fought', 'in', '1914'],
      ['There', 'were', 'over', 'a', 'million', 'deaths']),

     (['When', 'was', 'the', 'first', 'World', 'War', '?'],
      ['It', 'was', 'fought', 'in', '1914'],
      ['The', 'first', 'world', 'war', 'was', 'bad'])
    ]

    """
    if is_iterable:
        while True:
            j=0
            for q, doc, label in zip(queries, docs, labels):
                doc, label = (list(t) for t in zip(*sorted(zip(doc, label), reverse=True)))
                for item in zip(doc, label):
                    if item[1] == 1:
                        for new_item in zip(doc, label):
                            if new_item[1] == 0:
                                j+=1
                                yield(_make_indexed(q), _make_indexed(item[0]), _make_indexed(new_item[0]))
            print("SAMPLA RE!!!!!!!!!!!!!!!!!!", j)
    else:
        for q, doc, label in zip(queries, docs, labels):
            doc, label = (list(t) for t in zip(*sorted(zip(doc, label), reverse=True)))
            for item in zip(doc, label):
                if item[1] == 1:
                    for new_item in zip(doc, label):
                        if new_item[1] == 0:
                            yield(_make_indexed(q), _make_indexed(item[0]), _make_indexed(new_item[0]))


class BiDAF(utils.SaveLoad):
    """Model for training a Similarity Learning Model using the BiDAF model.
    You only have to provide sentences in the data as a list of words.
    """

    def __init__(self, queries=None, docs=None, labels=None, word_embedding=None,
                 text_maxlen=200, normalize_embeddings=True, epochs=10, unk_handle_method='random',
                 validation_data=None, topk=50, target_mode='ranking', verbose=1, batch_size=10):
        """Initializes the model and trains it

        Parameters
        ----------
        queries: iterable list of list of string words, optional
            The questions for the similarity learning model.
        docs: iterable list of list of list of string words, optional
            The candidate answers for the similarity learning model.
        labels: iterable list of list of int, optional
            Indicates when a candidate document is relevant to a query
            - 1 : relevant
            - 0 : irrelevant
        word_embedding : :class:`~gensim.models.keyedvectors.KeyedVectors`, optional
            a KeyedVector object which has the embeddings pre-loaded.
            If None, random word embeddings will be used.
        text_maxlen : int, optional
            The maximum possible length of a query or a document.
            This is used for padding sentences.
        normalize_embeddings : bool, optional
            Whether the word embeddings provided should be normalized.
        epochs : int, optional
            The number of epochs for which the model should train on the data.
        unk_handle_method : {'zero', 'random'}, optional
            The method for handling unkown words.
                - 'zero' : unknown words are given a zero vector
                - 'random' : unknown words are given a uniformly random vector bassed on the word string hash
        validation_data: list of the form [test_queries, test_docs, test_labels], optional
            where test_queries, test_docs  and test_labels are of the same form as
            their counter parts stated above.
        topk : int, optional
            the k topmost values in the interaction matrix between the queries and the docs
        target_mode : {'ranking', 'classification'}, optional
            the way the model should be trained, either to rank or classify
        verbose : {0, 1, 2}
            the level of information shared while training
                - 0 : silent
                - 1 : progress bar
                - 2 : one line per epoch


        Examples
        --------
        The trained model needs to be trained on data in the format

        >>> queries = ["When was World War 1 fought ?".lower().split(), "When was Gandhi born ?".lower().split()]
        >>> docs = [["The world war was bad".lower().split(), "It was fought in 1996".lower().split()], ["Gandhi was"
        ...    "born in the 18th century".lower().split(), "He fought for the Indian freedom movement".lower().split(),
        ...    "Gandhi was assasinated".lower().split()]]
        >>> labels = [[0, 1], [1, 0, 0]]
        >>> import gensim.downloader as api
        >>> word_embeddings_kv = api.load('glove-wiki-gigaword-50')
        >>> model = DRMM_TKS(queries, docs, labels, word_embedding=word_embeddings_kv, verbose=0)
        """
        self.queries = queries
        self.docs = docs
        self.labels = labels
        self.word_counter = Counter()
        self.text_maxlen = text_maxlen
        self.topk = topk
        self.word_embedding = word_embedding
        self.word2index, self.index2word = {}, {}
        self.normalize_embeddings = normalize_embeddings
        self.model = None
        self.epochs = epochs
        self.validation_data = validation_data
        self.target_mode = target_mode
        self.verbose = verbose
        self.first_train = True  # Whether the model has been trained before
        self.needs_vocab_build = True
        self.batch_size = batch_size
        self.max_passage_sents = 30

        # These functions have been defined outside the class and set as attributes here
        # so that they can be ignored when saving the model to file
        self._get_pair_list = _get_pair_list
        self._get_full_batch_iter = _get_full_batch_iter

        if self.target_mode not in ['ranking', 'classification']:
            raise ValueError(
                "Unkown target_mode %s. It must be either 'ranking' or 'classification'" % self.target_mode
            )

        if unk_handle_method not in ['random', 'zero']:
            raise ValueError("Unkown token handling method %s" % str(unk_handle_method))
        self.unk_handle_method = unk_handle_method

        if self.queries is not None and self.docs is not None and self.labels is not None:
            self.build_vocab(self.queries, self.docs, self.labels, self.word_embedding)
            self.train(self.queries, self.docs, self.labels, self.word_embedding,
                       self.text_maxlen, self.normalize_embeddings, self.epochs, self.unk_handle_method,
                       self.validation_data, self.topk, self.target_mode, self.verbose)

    def build_vocab(self, queries, docs, labels, word_embedding):
        """Indexes all the words and makes an embedding_matrix which
        can be fed directly into an Embedding layer
        """

        logger.info("Starting Vocab Build")

        # get all the vocab words
        for q in self.queries:
            self.word_counter.update(q)
        for doc in self.docs:
            for d in doc:
                self.word_counter.update(d)
        for i, word in enumerate(self.word_counter.keys()):
            self.word2index[word] = i
            self.index2word[i] = word

        self.vocab_size = len(self.word2index)
        logger.info("Vocab Build Complete")
        logger.info("Vocab Size is %d", self.vocab_size)

        logger.info("Building embedding index using KeyedVector pretrained word embeddings")
        if type(self.word_embedding) == KeyedVectors:
            kv_model = self.word_embedding
            embedding_vocab_size, self.embedding_dim = len(kv_model.vocab), kv_model.vector_size
        else:
            raise ValueError(
                    "Unknown value of word_embedding : %s. Must be either a KeyedVector object",
                    str(word_embedding)
                )

        logger.info(
            "The embeddings_index built from the given file has %d words of %d dimensions",
            embedding_vocab_size, self.embedding_dim
        )

        logger.info("Building the Embedding Matrix for the model's Embedding Layer")

        # Initialize the embedding matrix
        # UNK word gets the vector based on the method
        if self.unk_handle_method == 'random':
            self.embedding_matrix = np.random.uniform(-0.2, 0.2, (self.vocab_size, self.embedding_dim))
        elif self.unk_handle_method == 'zero':
            self.embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim))

        n_non_embedding_words = 0
        for word, i in self.word2index.items():
            if word in kv_model:
                # words not found in keyed vectors will get the vector based on unk_handle_method
                self.embedding_matrix[i] = kv_model[word]
            else:
                if self.unk_handle_method == 'random':
                    # Creates the same random vector for the given string each time
                    self.embedding_matrix[i] = self._seeded_vector(word, self.embedding_dim)
                n_non_embedding_words += 1
        logger.info(
            "There are %d words out of %d (%.2f%%) not in the embeddings. Setting them to %s", n_non_embedding_words,
            self.vocab_size, n_non_embedding_words * 100 / self.vocab_size, self.unk_handle_method
        )

        # Include embeddings for words in embedding file but not in the train vocab
        # It will be useful for embedding words encountered in validation and test set
        logger.info(
            "Adding additional words from the embedding file to embedding matrix"
        )

        # The point where vocab words end
        vocab_offset = self.vocab_size
        extra_embeddings = []
        # Take the words in the embedding file which aren't there int the train vocab
        for word in list(kv_model.vocab):
            if word not in self.word2index:
                # Add the new word's vector and index it
                extra_embeddings.append(kv_model[word])
                # We also need to keep an additional indexing of these
                # words
                self.word2index[word] = vocab_offset
                vocab_offset += 1

        # Set the pad and unk word to second last and last index
        self.pad_word_index = vocab_offset
        self.unk_word_index = vocab_offset + 1

        if self.unk_handle_method == 'random':
            unk_embedding_row = np.random.uniform(-0.2, 0.2, (1, self.embedding_dim))
        elif self.unk_handle_method == 'zero':
            unk_embedding_row = np.zeros((1, self.embedding_dim))

        pad_embedding_row = np.random.uniform(-0.2,
                                              0.2, (1, self.embedding_dim))

        if len(extra_embeddings) > 0:
            self.embedding_matrix = np.vstack(
                [self.embedding_matrix, np.array(extra_embeddings),
                 pad_embedding_row, unk_embedding_row]
            )
        else:
            self.embedding_matrix = np.vstack(
                [self.embedding_matrix, pad_embedding_row, unk_embedding_row]
            )

        if self.normalize_embeddings:
            logger.info("Normalizing the word embeddings")
            self.embedding_matrix = normalize(self.embedding_matrix)

        logger.info("Embedding Matrix build complete. It now has shape %s", str(self.embedding_matrix.shape))
        logger.info("Pad word has been set to index %d", self.pad_word_index)
        logger.info("Unknown word has been set to index %d", self.unk_word_index)
        logger.info("Embedding index build complete")
        self.needs_vocab_build = False

    def _string2numeric_hash(self, text):
        "Gets a numeric hash for a given string"
        return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)

    def _seeded_vector(self, seed_string, vector_size):
        """Create one 'random' vector (but deterministic by seed_string)"""
        # Note: built-in hash() may vary by Python version or even (in Py3.x) per launch
        once = np_random.RandomState(self._string2numeric_hash(seed_string) & 0xffffffff)
        return (once.rand(vector_size) - 0.5) / vector_size

    def _make_indexed(self, sentence):
        """Gets the indexed version of the sentence based on the self.word2index dict
        in the form of a list

        This function should never encounter any OOV words since it only indexes
        in vocab words

        Parameters
        ----------
        sentence : iterable list of list of str
            The sentence to be indexed

        Raises
        ------
        ValueError : If the sentence has a lenght more than text_maxlen
        """

        indexed_sent = []
        for word in sentence:
            if word in self.word2index:
                indexed_sent.append(self.word2index[word])
            else:
                indexed_sent.append(self.unk_word_index)

        if len(indexed_sent) > self.text_maxlen:
            indexed_sent = indexed_sent[:self.text_maxlen]

            # print(
            #     "text_maxlen: %d isn't big enough. Error at sentence of length %d."
            #     "Sentence is %s" % (self.text_maxlen, len(sentence), sentence)
            # )
        indexed_sent = indexed_sent + [self.pad_word_index] * (self.text_maxlen - len(indexed_sent))
        return indexed_sent

    def _get_full_batch(self):
        """Provides all the data points int the format: X1, X2, y with
        alternate positive and negative examples

        Returns
        -------
        X1 : numpy array of shape (num_samples, text_maxlen)
            the queries
        X2 : numpy array of shape (num_samples, text_maxlen)
            the docs
        y : numpy array with {0, 1} of shape (num_samples,)
            The relation between X1[i] and X2[j]
            1 : X2[i] is relevant to X1[i]
            0 : X2[i] is not relevant to X1[i]
        """
        X1, X2, y = [], [], []
        for i, (query, pos_doc, neg_doc) in enumerate(self.pair_list):
            X1.append(query)
            X2.append(pos_doc)
            y.append(1)
            X1.append(query)
            X2.append(neg_doc)
            y.append(0)

        print('There are pairs in pair_list', np.array(X1).shape, np.array(X2).shape, np.array(y).shape)
        return np.array(X1), np.array(X2), np.array(y)

    def new_full_batch_iter(self, batch_size):
        train_queries = []
        train_docs = []
        train_labels = []

        batch_q, batch_d, batch_l = [], [], []

        while True:
            i = 0
            for q, doc, label in zip(self.queries, self.docs, self.labels):
                train_queries.append(self._make_indexed(q))
                for d, l  in zip(doc, label):
                    train_docs.append(self._make_indexed(d))
                    
                    if label == 0:
                        train_labels.append([1, 0])
                    else:
                        train_labels.append([0, 1])
                if len(train_docs) <= self.max_passage_sents:
                    while(len(train_docs) != self.max_passage_sents):
                        train_docs.append([self.pad_word_index]*self.text_maxlen)
                        train_labels.append([1, 0])
                else:
                    raise ValueError('max_passage_sents is less than ' + str(len(train_docs)))

                batch_q.append(train_queries)
                batch_d.append(train_docs)
                batch_l.append(train_labels)
                i += 1

                train_queries, train_docs, train_labels = [], [], []

                if i%batch_size == 0 and i!=0:
                    a, b, c = np.array(batch_q), np.array(batch_d), np.array(batch_l)
                    a = a.squeeze()
                    b = b.reshape((self.batch_size, self.max_passage_sents*self.text_maxlen))               
                    yield ({'question_input':a, 'passage_input':b}, c )
                    batch_q, batch_d, batch_l = [], [], []





    def train(self, queries, docs, labels, word_embedding=None,
              text_maxlen=200, normalize_embeddings=True, epochs=10, unk_handle_method='zero',
              validation_data=None, topk=20, target_mode='ranking', verbose=1, batch_size=5, steps_per_epoch=86):
        """Trains a DRMM_TKS model using specified parameters

        This method is called from on model initialization if the data is provided.
        It can also be trained in an online manner or after initialization
        """

        self.queries = queries or self.queries
        self.docs = docs or self.docs
        self.labels = labels or self.labels

        # This won't change the embedding layer TODO
        self.word_embedding = word_embedding or self.word_embedding
        self.text_maxlen = text_maxlen or self.text_maxlen
        self.normalize_embeddings = normalize_embeddings or self.normalize_embeddings
        self.epochs = epochs or self.epochs
        self.unk_handle_method = unk_handle_method or self.unk_handle_method
        self.validation_data = validation_data or self.validation_data
        self.topk = topk or self.topk
        self.target_mode = target_mode or self.target_mode

        if verbose != 0:  # Check needed since 0 or 2 will always give 2
            self.verbose = verbose or self.verbose
        else:
            self.verbose = 0

        if self.queries is None or self.docs is None or self.labels is None:
            raise ValueError("queries, docs and labels have to be specified")
        # We need to build these each time since any of the parameters can change from each train to trian
        if self.needs_vocab_build:
            self.build_vocab(self.queries, self.docs, self.labels, self.word_embedding)

        is_iterable = False
        if isinstance(self.queries, Iterable) and not isinstance(self.queries, list):
            is_iterable = True
            logger.info("Input is an iterable amd will be streamed")

        self.pair_list = self._get_pair_list(self.queries, self.docs, self.labels, self._make_indexed, is_iterable)
        if is_iterable:
            train_generator = self.new_full_batch_iter(self.batch_size)
        else:
            raise ValueError('Needs to be iterable')
            X1_train, X2_train, y_train = self._get_full_batch()

        # for i, j, k in train_generator:
        #     print(i, i.shape)
        #     print(j, j.shape)
        #     print(k, k.shape)
        #     exit()

        if self.first_train:
            # The settings below should be set only once
            self.model = self._get_keras_model()
            #from keras.optimizers import Adadelta

            self.model.compile(optimizer='adam', loss='categorical_crossentropy')

            # optimizer = 'adam'
            # optimizer = 'adadelta'
            # optimizer = optimizers.get(optimizer)
            # learning_rate = 0.0001
            # learning_rate = 1
            # K.set_value(optimizer.lr, learning_rate)
            # # either one can be selected. Currently, the choice is manual.
            # loss = hinge
            # loss = 'mse'
            # loss = rank_hinge_loss
            # self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        else:
            logger.info("Model will be retrained")

        self.model.summary()

        # Put the validation data in as a callback
        val_callback = None
        if self.validation_data is not None:
            test_queries, test_docs, test_labels = self.validation_data

            long_doc_list = []
            long_label_list = []
            long_query_list = []
            doc_lens = []

            for query, doc, label in zip(test_queries, test_docs, test_labels):
                i = 0
                for d, l in zip(doc, label):
                    long_query_list.append(query)
                    long_doc_list.append(d)
                    long_label_list.append(l)
                    i += 1
                doc_lens.append(len(doc))

            indexed_long_query_list = self._translate_user_data(long_query_list)
            indexed_long_doc_list = self._translate_user_data(long_doc_list)

            val_callback = ValidationCallback(
                                {"X1": indexed_long_query_list, "X2": indexed_long_doc_list, "doc_lengths": doc_lens,
                                "y": long_label_list}
                            )
            val_callback = [val_callback]  # since `model.fit` requires a list

        # If train is called again, not all values should be reset
        if self.first_train is True:
            self.first_train = False

        if is_iterable:
            self.model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, callbacks=val_callback,
                                    epochs=self.epochs, shuffle=False)
        else:
            self.model.fit(x={"query": X1_train, "doc": X2_train}, y=y_train, batch_size=5,
                           verbose=self.verbose, epochs=self.epochs, shuffle=False, callbacks=val_callback)

    def _translate_user_data(self, data):
        """Translates given user data into an indexed format which the model understands.
        If a model is not in the vocabulary, it is assigned the `unk_word_index` which maps
        to the unk vector decided by `unk_handle_method`

        Parameters
        ----------
        data : list of list of string words
            The data to be tranlsated

        Examples
        --------
        >>> from gensim.test.utils import datapath
        >>> model = DRMM_TKS.load(datapath('drmm_tks'))
        >>>
        >>> queries = ["When was World War 1 fought ?".split(), "When was Gandhi born ?".split()]
        >>> print(model._translate_user_data(queries))
        [[31  1 23 31  4  5  6 30 30 30]
         [31  1 31  8  6 30 30 30 30 30]]
        """
        translated_data = []
        n_skipped_words = 0
        for sentence in data:
            translated_sentence = []
            for word in sentence:
                if word in self.word2index:
                    translated_sentence.append(self.word2index[word])
                else:
                    # If the key isn't there give it the zero word index
                    translated_sentence.append(self.unk_word_index)
                    n_skipped_words += 1
            if len(sentence) > self.text_maxlen:
                translated_sentence = translated_sentence[:self.text_maxlen]
                # logger.info(
                #     "text_maxlen: %d isn't big enough. Error at sentence of length %d."
                #     "Sentence is %s", self.text_maxlen, len(sentence), str(sentence)
                # )
            translated_sentence = translated_sentence + \
                (self.text_maxlen - len(sentence)) * [self.pad_word_index]
            translated_data.append(np.array(translated_sentence))

        logger.info(
            "Found %d unknown words. Set them to unknown word index : %d", n_skipped_words, self.unk_word_index
        )
        return np.array(translated_data)

    def predict(self, queries, docs, labels):
        """Predcits the similarity between a query-document pair
        based on the trained DRMM TKS model

        Parameters
        ----------
        queries : list of list of str
            The questions for the similarity learning model
        docs : list of list of list of str
            The candidate answers for the similarity learning model


        Examples
        --------
        >>> from gensim.test.utils import datapath
        >>> model = DRMM_TKS.load(datapath('drmm_tks'))
        >>>
        >>> queries = ["When was World War 1 fought ?".split(), "When was Gandhi born ?".split()]
        >>> docs = [["The world war was bad".split(), "It was fought in 1996".split()], ["Gandhi was born in the 18th"
        ...        " century".split(), "He fought for the Indian freedom movement".split(), "Gandhi was"
        ...        " assasinated".split()]]
        >>> print(model.predict(queries, docs))
        [[0.9933108 ]
         [0.9925415 ]
         [0.9827911 ]
         [0.99258184]
         [0.9960481 ]]
        """
        train_queries = []
        train_docs = []
        train_labels = []
        doc_lens = []

        batch_q, batch_d, batch_l = [], [], []

        i = 0
        for q, doc, label in zip(queries, docs, labels):
            train_queries.append(self._make_indexed(q))
            for d, l  in zip(doc, label):
                train_docs.append(self._make_indexed(d))
                
                if label == 0:
                    train_labels.append([1, 0])
                else:
                    train_labels.append([0, 1])
            doc_lens.append(len(train_docs))
            if len(train_docs) <= self.max_passage_sents:
                while(len(train_docs) != self.max_passage_sents):
                    train_docs.append([self.pad_word_index]*self.text_maxlen)
                    train_labels.append([1, 0])
            else:
                raise ValueError('max_passage_sents is less than ' + str(len(train_docs)))

            batch_q.append(train_queries)
            batch_d.append(train_docs)
            batch_l.append(train_labels)
            i += 1

            train_queries, train_docs, train_labels = [], [], []

        a, b, c = np.array(batch_q), np.array(batch_d), np.array(batch_l)
        a = a.squeeze()
        b = b.reshape((-1, self.max_passage_sents*self.text_maxlen))                
        print(self.model.evaluate({'question_input':a, 'passage_input':b}, c))
        preds = self.model.predict({'question_input':a, 'passage_input':b})
        final_preds = []
        for pred, d_len in zip(preds, doc_lens):
            if d_len > 0:
                for p in pred[:d_len]:
                    final_preds.append(p)
        #print(final_preds, np.array(final_preds).shape)
        batch_q, batch_d, batch_l = [], [], []

        '''
        long_query_list = []
        long_doc_list = []
        for query, doc in zip(queries, docs):
            long_query_list.append(query)
            for d in doc:
                long_doc_list.append(d)

        indexed_long_query_list = self._translate_user_data(long_query_list)
        indexed_long_doc_list = self._translate_user_data(long_doc_list)

        predictions = self.model.predict(x={'query': indexed_long_query_list, 'doc': indexed_long_doc_list})

        logger.info("Predictions in the format query, doc, similarity")
        for i, (q, d) in enumerate(zip(long_query_list, long_doc_list)):
            logger.info("%s\t%s\t%s", str(q), str(d), str(predictions[i][0]))

        return predictions
        '''

    def tiny_predict(self, q, d):
        q = self._make_indexed(q)
        train_docs = []
        train_docs.append(self._make_indexed(d))

        if len(train_docs) <= self.max_passage_sents:
            while(len(train_docs) != self.max_passage_sents):
                train_docs.append([self.pad_word_index]*self.text_maxlen)
        
        preds = self.model.predict({'question_input':a, 'passage_input':b})


    def evaluate(self, queries, docs, labels):
        """Evaluates the model and provides the results in terms of metrics (MAP, nDCG)
        This should ideally be called on the test set.

        Parameters
        ----------
        queries : list of list of str
            The questions for the similarity learning model
        docs : list of list of list of str
            The candidate answers for the similarity learning model
        labels : list of list of int
            The relevance of the document to the query. 1 = relevant, 0 = not relevant
        """
        long_doc_list = []
        long_label_list = []
        long_query_list = []
        doc_lens = []

        for query, doc, label in zip(queries, docs, labels):
            i = 0
            for d, l in zip(doc, label):
                long_query_list.append(query)
                long_doc_list.append(d)
                long_label_list.append(l)
                i += 1
            doc_lens.append(len(doc))
        indexed_long_query_list = self._translate_user_data(long_query_list)
        indexed_long_doc_list = self._translate_user_data(long_doc_list)
        predictions = self.model.predict(x={'query': indexed_long_query_list, 'doc': indexed_long_doc_list})
        Y_pred = []
        Y_true = []
        offset = 0
        for doc_size in doc_lens:
            Y_pred.append(predictions[offset: offset + doc_size])
            Y_true.append(long_label_list[offset: offset + doc_size])
            offset += doc_size
        logger.info("MAP: %.2f", mapk(Y_true, Y_pred))
        for k in [1, 3, 5, 10, 20]:
            logger.info("nDCG@%d : %.2f", k, mean_ndcg(Y_true, Y_pred, k=k))

    def save(self, fname, *args, **kwargs):
        """Save the model.
        This saved model can be loaded again using :func:`~gensim.models.experimental.drmm_tks.DRMM_TKS.load`
        The keras model shouldn't be serialized using pickle or cPickle. So, the non-keras
        variables will be saved using gensim's SaveLoad and the keras model will be saved using
        the keras save method with ".keras" prefix.

        Also see :func:`~gensim.models.experimental.drmm_tks.DRMM_TKS.load`

        Parameters
        ----------
        fname : str
            Path to the file.

        Examples
        --------
        >>> from gensim.test.utils import datapath, get_tmpfile
        >>> model = DRMM_TKS.load(datapath('drmm_tks'))
        >>> model_save_path = get_tmpfile('drmm_tks_model')
        >>> model.save(model_save_path)
        """
        # don't save the keras model as it needs to be saved with a keras function
        # Also, we can't save iterable properties. So, ignore them.
        kwargs['ignore'] = kwargs.get(
                            'ignore', ['model', '_get_pair_list', '_get_full_batch_iter',
                                        'queries', 'docs', 'labels', 'pair_list'])
        kwargs['fname_or_handle'] = fname
        super(BiDAF, self).save(*args, **kwargs)
        self.model.save(fname + ".keras")

    @classmethod
    def load(cls, *args, **kwargs):
        """Loads a previously saved `DRMM TKS` model. Also see `save()`.
        Collects the gensim and the keras models and returns it as on gensim model.

        Parameters
        ----------
        fname : str
            Path to the saved file.

        Returns
        -------
        :obj: `~gensim.models.experimental.DRMM_TKS`
            Returns the loaded model as an instance of :class: `~gensim.models.experimental.DRMM_TKS`.


        Examples
        --------
        >>> from gensim.test.utils import datapath, get_tmpfile
        >>> model_file_path = datapath('drmm_tks')
        >>> model = DRMM_TKS.load(model_file_path)
        """
        fname = args[0]
        gensim_model = super(DRMM_TKS, cls).load(*args, **kwargs)
        keras_model = load_model(
            fname + '.keras')
        gensim_model.model = keras_model
        gensim_model._get_pair_list = _get_pair_list
        gensim_model._get_full_batch_iter = _get_full_batch_iter
        return gensim_model

    def _get_keras_model(self, embed_trainable=False, dropout_rate=0.5, hidden_sizes=[100, 1]):

        max_question_words = self.text_maxlen
        num_passage_words = self.text_maxlen
        max_passage_sents = self.max_passage_sents
        num_highway_layers = 2
        highway_activation = 'relu'
        num_hidden_ending_bidir_layers = 2
        embedding_dim = 100
        n_encoder_hidden_nodes = 100
        max_passage_words = num_passage_words*max_passage_sents

        question_input = Input(shape=(max_question_words,), dtype='int32', name="question_input")
        passage_input = Input(shape=(max_passage_words,), dtype='int32', name="passage_input")

        #embedding_layer = Embedding(input_dim=70, output_dim=embedding_dim)
        embedding_layer = Embedding(self.embedding_matrix.shape[0], self.embedding_matrix.shape[1],
                              weights=[self.embedding_matrix], trainable=embed_trainable)

        question_embedding = embedding_layer(question_input)  # (None, max_question_words, embedding_dim)
        passage_embedding = embedding_layer(passage_input)  # (None, max_passage_words*max_passage_sents, embedding_dim)

        for i in range(num_highway_layers):
            highway_layer = Highway(activation=highway_activation, name='highway_{}'.format(i))
            
            question_layer = TimeDistributed(highway_layer, name=highway_layer.name + "_qtd")
            question_embedding = question_layer(question_embedding)

            passage_layer = TimeDistributed(highway_layer, name=highway_layer.name + "_ptd")
            passage_embedding = passage_layer(passage_embedding)


        passage_bidir_encoder = Bidirectional(LSTM(n_encoder_hidden_nodes, return_sequences=True,
                                                                   name='PassageBidirEncoder'), merge_mode='concat')

        encoded_passage = passage_bidir_encoder(passage_embedding)
        encoded_question = passage_bidir_encoder(question_embedding)
        tiled_passage = Lambda(lambda x: tf.tile(tf.expand_dims(x, 2), [1, 1, max_question_words, 1]))(encoded_passage)
        tiled_question = Lambda(lambda x: tf.tile(tf.expand_dims(x, 1), [1, max_passage_words, 1, 1]))(encoded_question)

        # (batch_size, max_passage_sents, max_question_words, 2*n_encoder_hidden_nodes)
        a_elmwise_mul_b = Lambda(lambda x:tf.multiply(x[0], x[1]))([tiled_passage, tiled_question])

        # (batch_size, max_passage_sents, max_question_words, 6*n_encoder_hidden_nodes)
        cat_data = Concatenate()([tiled_passage, tiled_question, a_elmwise_mul_b])

        S = Dense(1)(cat_data)
        S = Lambda(lambda x: K.squeeze(x, -1))(S)  # (batch_size, max_passage_sents, max_question_words)

        S = Activation('softmax')(S)

        c2q = Lambda(lambda x: tf.matmul(x[0], x[1]))([S, encoded_question]) # (N, T, 2d) = bmm( (N, T, J), (N, J, 2d) )

        # Query2Context
        # b: attention weights on the context
        b = Lambda(lambda x: tf.nn.softmax(K.max(x, 2), dim=-1), name='b')(S) # (N, T)

        q2c = Lambda(lambda x:tf.matmul(tf.expand_dims(x[0], 1), x[1]))([b, encoded_passage]) # (N, 1, 2d) = bmm( (N, 1, T), (N, T, 2d) )
        q2c = Lambda(lambda x: tf.tile(x, [1, max_passage_words, 1]))(q2c) # (N, T, 2d), tiled T times


        # G: query aware representation of each context word
        G = Lambda(lambda x: tf.concat([x[0], x[1], tf.multiply(x[0], x[1]), tf.multiply(x[0], x[2])], axis=2)) ([encoded_passage, c2q, q2c]) # (N, T, 8d)


        modelled_passage = Bidirectional(LSTM(n_encoder_hidden_nodes, return_sequences=True))(G)
        modelled_passage = Bidirectional(LSTM(n_encoder_hidden_nodes, return_sequences=True))(modelled_passage)

        # Reshape it back to be at the sentence level
        reshaped_passage = Reshape((max_passage_sents, num_passage_words, n_encoder_hidden_nodes*2))(modelled_passage)
        g2 = Lambda(lambda x: tf.reduce_sum(x, 2))(reshaped_passage)
        # g2_ = Reshape([n_encoder_hidden_nodes*2])(g2)
        pred = Dense(2, activation='softmax')(g2)

        model = Model(inputs=[question_input, passage_input], outputs=[pred])
        return model

    def tiny_predict(self, q, d):
        """To make a prediction on on query and a doc

        Parameters
        ----------
        q : str
        d : str
        """
        q = [self._make_indexed(q)]
        train_docs = []
        train_docs.append(self._make_indexed(d))

        if len(train_docs) <= self.max_passage_sents:
            while(len(train_docs) != self.max_passage_sents):
                train_docs.append([self.pad_word_index]*self.text_maxlen)
        q = np.array(q).reshape((1, self.text_maxlen))
        train_docs = np.array(train_docs).reshape((1, self.text_maxlen*self.max_passage_sents))
        #print(q, q.shape)
        #print(train_docs, train_docs.shape) 
        # q = q.reshape((self.text_maxlen))
        preds = self.model.predict(x={'question_input':q,  'passage_input':train_docs})
        #print(preds)
        return preds[0][0][1]


    def batch_tiny_predict(self, q, doc):
        """To make a prediction on on query and a batch of docs
        Typically speeds up prediction

        Parameters
        ----------
        q : str
        d : list of str
        """
        q = [self._make_indexed(q)]
        train_docs = []
        for d in doc:
            train_docs.append(self._make_indexed(d))

        d_len = len(train_docs)

        if len(train_docs) <= self.max_passage_sents:
            while(len(train_docs) != self.max_passage_sents):
                train_docs.append([self.pad_word_index]*self.text_maxlen)
        q = np.array(q).reshape((1, self.text_maxlen))
        train_docs = np.array(train_docs).reshape((1, self.text_maxlen*self.max_passage_sents))
        #print(q, q.shape)
        #print(train_docs, train_docs.shape) 
        # q = q.reshape((self.text_maxlen))
        preds = self.model.predict(x={'question_input':q,  'passage_input':train_docs})
        #print(preds)
        return preds[0][:d_len]
   
    '''
    def true_batch_tiny_predict(self, queries, docs, batch_size):
        for query in queries:
        q = [self._make_indexed(q)]
        train_docs = []
        for d in doc:
            train_docs.append(self._make_indexed(d))

        d_len = len(train_docs)

        if len(train_docs) <= self.max_passage_sents:
            while(len(train_docs) != self.max_passage_sents):
                train_docs.append([self.pad_word_index]*self.text_maxlen)
        q = np.array(q).reshape((1, self.text_maxlen))
        train_docs = np.array(train_docs).reshape((1, self.text_maxlen*self.max_passage_sents))
        #print(q, q.shape)
        #print(train_docs, train_docs.shape) 
        # q = q.reshape((self.text_maxlen))
        preds = self.model.predict(x={'question_input':q,  'passage_input':train_docs})
        #print(preds)
        return preds[0][:d_len]'''
