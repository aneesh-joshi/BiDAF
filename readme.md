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


## Disclaimer
Code has been reused from old DRMM TKS code. So docstrings are faulty. The old code is only responsible for word indexing, etc.