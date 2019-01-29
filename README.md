# Hierarchical Recurrent Encoder-Decoder code (HRED) for Query Suggestion.

This code is tensorflow implementation of the paper:

["A Hierarchical Recurrent Encoder-Decoder For Generative Context-Aware Query Suggestion", by Alessandro Sordoni, Yoshua Bengio, Hossein Vahabi, Christina Lioma, Jakob G. Simonsen, Jian-Yun Nie, to appear in CIKM'15.](http://arxiv.org/abs/1507.02221)

Extension of original [repo by Beomsu Kim](https://github.com/shuuki4/HRED-tensorflow)

## Setup
```
pip install -r requirements.txt
```

## Sample data generation

```
python gen_sample_data.py
```
This will create the preprocessed dataset for training.

## Training

```
./train.sh
```

## Inference

```
./infer.sh
```
