# Named Entity Recognition (NER) using BiLSTM CRF
This is a Pytorch implementation of BiLSTM-CRF for Named Entity Recognition, which is described in [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991)

## Data
The corpus in the [data](./data) folder is MSRA Chinese NER corpus. Since there are no development data, we split the data in train.txt into training and development part when traing the model.

## Usage
For training the model, you can use the following command:
```
sh run.sh train
```
For those who are not able to use GPU, use the following command to train:
```
sh run.sh train-without-cuda
```
For testing, you can use the following command:
```
sh run.sh test
```
Also, if you are not able to use GPU, you can use (this procedure won't take long time when using CPU):
```
sh run.sh test-without-cuda
```
There is already a trained model in the [model](./model) folder, so you can directly executing the testing command directly without training.

If you want to change some hyper-parameters, use the following command to refer to the options.
```
python run.py --help
```

## Result
We evaluate the model by F1 score, here is an example for computing `TP`, `FP` and `FN`:
```
True tag: ['O', 'B-PER', 'I-PER', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'O']
Predicted tag: ['O', 'B-PER', 'I-PER', 'O', 'O', 'B-ORG', 'I-ORG', 'O', 'O']
```
We have TP=1 (`['B-PER', 'I-PER']` in both tags), FN=1(`['B-ORG', 'I-ORG', 'I-ORG']` in the true tag), FP=1(`['B-ORG', 'I-ORG']` in the predicted tag).

The experiment result on testing data of the trained model is as follows:
 * Precison: 0.929792
 * Recall: 0.920935
 * F1 score: 0.925342

## Reference
  1. [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991)
  2. [cs224n Assignment 4](http://web.stanford.edu/class/cs224n/index.html#schedule)
