# Interactive masked language model (MLM) with pre-trained BERT models

## Requirements

[pytorch 1.0](https://pytorch.org)

[pytorch-pretrained-bert](https://github.com/huggingface/pytorch-pretrained-bert)

## Download data

`bash ./download.sh`

By default, it will download `bert-base-uncased` vocabulary and pre-trained weights.

## Run

```python
python3 interactive_bert.py
```

## Some examples

```
Initialize BERT vocabulary from bert_models/bert-base-uncased-vocab.txt...
Initialize BERT model from bert_models/bert-base-uncased.tar.gz...

>>> Enter your message: I would like to have some [MASK] for lunch .
Top 5 predictions for 1th [MASK]:
coffee 0.05741667374968529
food 0.04030166566371918
company 0.037628173828125
pancakes 0.03524628281593323
fish 0.033470142632722855
================================================================================

>>> Enter your message: united states is famous for [MASK] .
Top 5 predictions for 1th [MASK]:
tourism 0.029551107436418533
fishing 0.0217527337372303
music 0.013582928106188774
agriculture 0.011571155861020088
football 0.010294185020029545
================================================================================

>>> Enter your message: please welcome president [MASK] .
Top 5 predictions for 1th [MASK]:
johnson 0.06361229717731476
kennedy 0.060998450964689255
obama 0.03890557959675789
lincoln 0.028362315148115158
wilson 0.015903476625680923
================================================================================
```
