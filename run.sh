#!/bin/sh

if [ "$1" = "train" ]
then
  python run.py train ./data/train.txt ./vocab/sent_vocab.json ./vocab/tag_vocab.json --cuda
elif [ "$1" = "train-without-cuda" ]
then
  python run.py train ./data/train.txt ./vocab/sent_vocab.json ./vocab/tag_vocab.json
elif [ "$1" = "test" ]
then
  python run.py test ./data/test.txt ./result.txt ./vocab/sent_vocab.json ./vocab/tag_vocab.json ./model/model.pth --cuda
  perl conlleval.pl < result.txt
elif [ "$1" = "test-without-cuda" ]
then
  python run.py test ./data/test.txt ./result.txt  ./vocab/sent_vocab.json ./vocab/tag_vocab.json ./model/model.pth
  perl conlleval.pl < result.txt
elif [ "$1" = "vocab" ]
then
	python vocab.py ./data/train.txt ./vocab/sent_vocab.json ./vocab/tag_vocab.json
fi
