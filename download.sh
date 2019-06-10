#!/bin/bash

BERT_MODELS="./bert_models/"

if ! [[ -e ${BERT_MODELS}  ]]; then
    mkdir -p ${BERT_MODELS}
fi

cd ${BERT_MODELS}
wget -c https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt

wget -c https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz

cd ..

echo "Download data complete !"

