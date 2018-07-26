# BiDirectional Attention Flow for QA Transfer

This repo contains code which attempts to reproduce the [QA-Transfer Paper](http://aclweb.org/anthology/P17-2081) which makes of the BiDAF model introduced in [this](https://arxiv.org/abs/1611.01603) paper.

The BiDAF model has been largely adapted from (this implementation)[https://github.com/allenai/deep_qa/blob/master/deep_qa/models/reading_comprehension/bidirectional_attention.py] by AllenAI.

## Resources
- The QA Transfer tensorflow repo can be found [here](https://github.com/shmsw25/qa-transfer)
- The original bidaf code in tensorflow can be found [here](https://github.com/allenai/bi-att-flow)
- A pytorch implementation of BiDAF by AllenAI can be found [here](https://github.com/allenai/allennlp/blob/master/allennlp/models/reading_comprehension/bidaf.py)



The model pre-trains on SQUAD-T dataset. Then it is evaluated on the WikiQA test set.

The SQUAD-T data set has been extracted from the SQUAD dataset and is presented in this repo.

## To run my experiment
`$ python my_try.py`

The above script will run the bidaf model on SQUAD-T and evaluate it with WikiQA test set.


## Points of interest

- The model is defined in the function `_get_keras_model` in `bidaf.py`
- The training is done from the function `train` in `bidaf.py`

## General Explanation of what's going on
The model takes in a query and a passage.
The passage is made up of a sentences.
The model mostly treats the passage as a passage except when it comes to predicting, when it predicts on a per sentence basis.

While giving inputs, we give:


`(batch_size, max_query_words)` as the query.
Example:
`[['Hello', 'there'], ['I', 'am', 'Groot']] --PAD and INDEX--> [[12, 54, 101, 101, 101], [64, 23, 233, 101, 101]]  (2,5)`


`(batch_size, max_passage_words)` as the passage.
Here `max_passage_words = max_number_of_sents_per_question * max_length of a sentence`

So, every question will have a number of sentence as possible inputs. We pad each sentence to the same length.
We get: `(batch_size, max_sentences_per_query, max_length_of_each_sentence)`

Then we reshape the aboce to be just `(batch_size, max_sentences_per_query * max_length_of_each_sentence)`

So one sample may look like
`["This", "is", "the", "first", "sentence", "PAD", "PAD", 
  "The", "second", "one", "PAD", "PAD", "PAD", "PAD", 
  "PAD", "PAD", "PAD", "PAD", "PAD", "PAD", "PAD"]` # we need 3 sentences for the passage but have onlt 2, so we make an empty padding of the last one.

When we predict, we will reshape it to be on a per sentence basis again and predict from there.



## Disclaimer
Code has been reused from old DRMM TKS code. So docstrings are faulty. The old code is only responsible for word indexing, etc.

## Logs
Here are some logs on executing `my_try.py` (with Warnings removed)

```
Using TensorFlow backend.
max query, max number of docs per query and max number of docs
33 29 230
Average query, average number of docs per query and average number of docs
10.405203405865658 5.105676442762536 24.902959215817074
2018-07-26 11:37:50,933 : INFO : loading projection weights from /home/aneeshyjoshi/gensim-data/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz
2018-07-26 11:38:10,006 : INFO : loaded (400000, 50) matrix from /home/aneeshyjoshi/gensim-data/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz
2018-07-26 11:38:10,006 : INFO : Starting Vocab Build
2018-07-26 11:38:12,061 : INFO : Vocab Build Complete
2018-07-26 11:38:12,061 : INFO : Vocab Size is 23793
2018-07-26 11:38:12,061 : INFO : Building embedding index using KeyedVector pretrained word embeddings
2018-07-26 11:38:12,061 : INFO : The embeddings_index built from the given file has 400000 words of 50 dimensions
2018-07-26 11:38:12,061 : INFO : Building the Embedding Matrix for the model's Embedding Layer
2018-07-26 11:38:12,135 : INFO : There are 1523 words out of 23793 (6.40%) not in the embeddings. Setting them to zero
2018-07-26 11:38:12,135 : INFO : Adding additional words from the embedding file to embedding matrix
2018-07-26 11:38:13,291 : INFO : Normalizing the word embeddings
2018-07-26 11:38:13,469 : INFO : Embedding Matrix build complete. It now has shape (401525, 50)
2018-07-26 11:38:13,469 : INFO : Pad word has been set to index 401523
2018-07-26 11:38:13,469 : INFO : Unknown word has been set to index 401524
2018-07-26 11:38:13,470 : INFO : Embedding index build complete
2018-07-26 11:38:13,504 : INFO : Input is an iterable amd will be streamed
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
passage_input (InputLayer)      (None, 1530)         0                                            
__________________________________________________________________________________________________
question_input (InputLayer)     (None, 51)           0                                            
__________________________________________________________________________________________________
embedding_1 (Embedding)         multiple             20076250    question_input[0][0]             
                                                                 passage_input[0][0]              
__________________________________________________________________________________________________
highway_0_ptd (TimeDistributed) (None, 1530, 50)     5100        embedding_1[1][0]                
__________________________________________________________________________________________________
highway_0_qtd (TimeDistributed) (None, 51, 50)       5100        embedding_1[0][0]                
__________________________________________________________________________________________________
highway_1_ptd (TimeDistributed) (None, 1530, 50)     5100        highway_0_ptd[0][0]              
__________________________________________________________________________________________________
highway_1_qtd (TimeDistributed) (None, 51, 50)       5100        highway_0_qtd[0][0]              
__________________________________________________________________________________________________
bidirectional_1 (Bidirectional) multiple             120800      highway_1_ptd[0][0]              
                                                                 highway_1_qtd[0][0]              
__________________________________________________________________________________________________
passage_question_similarity (Ma (None, 1530, 51)     0           bidirectional_1[0][0]            
                                                                 bidirectional_1[1][0]            
__________________________________________________________________________________________________
max_1 (Max)                     (None, 1530)         0           passage_question_similarity[0][0]
__________________________________________________________________________________________________
activation_2 (Activation)       (None, 1530)         0           max_1[0][0]                      
__________________________________________________________________________________________________
activation_1 (Activation)       (None, 1530, 51)     0           passage_question_similarity[0][0]
__________________________________________________________________________________________________
question_passage_vector (Weight (None, 200)          0           bidirectional_1[0][0]            
                                                                 activation_2[0][0]               
__________________________________________________________________________________________________
passage_question_vectors (Weigh (None, 1530, 200)    0           bidirectional_1[1][0]            
                                                                 activation_1[0][0]               
__________________________________________________________________________________________________
repeat_like_1 (RepeatLike)      (None, 1530, 200)    0           question_passage_vector[0][0]    
                                                                 bidirectional_1[0][0]            
__________________________________________________________________________________________________
final_merged_passage (ComplexCo (None, 1530, 800)    0           bidirectional_1[0][0]            
                                                                 passage_question_vectors[0][0]   
                                                                 repeat_like_1[0][0]              
__________________________________________________________________________________________________
bidirectional_2 (Bidirectional) (None, 1530, 200)    720800      final_merged_passage[0][0]       
__________________________________________________________________________________________________
bidirectional_3 (Bidirectional) (None, 1530, 200)    240800      bidirectional_2[0][0]            
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 1530, 1000)   0           final_merged_passage[0][0]       
                                                                 bidirectional_3[0][0]            
__________________________________________________________________________________________________
reshape_1 (Reshape)             (None, 30, 51, 1000) 0           concatenate_1[0][0]              
__________________________________________________________________________________________________
max_2 (Max)                     (None, 30, 51)       0           reshape_1[0][0]                  
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 30, 2)        104         max_2[0][0]                      
==================================================================================================
Total params: 21,168,954
Trainable params: 1,092,704
Non-trainable params: 20,076,250
__________________________________________________________________________________________________
Epoch 1/3
 2/86 [..............................] - ETA: 18:12 - loss: 0.7371

```