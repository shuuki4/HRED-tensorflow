#!/bin/bash
python3 -m hred.train \
    --train_path=data/train.txt \
    --val_path=data/valid.txt \
    --vocab_path=data/vocab.txt \
    --save_dir=data/model \
    --word_embedding_dim=32 \
    --num_rnn_layers=4 \
    --num_bidi_layers=1 \
    --num_rnn_units=64 \
    --optimizer=adam \
    --initial_learning_rate=1e-3 \
    --start_decay_step=10000 \
    --lr_decay_steps=5000 \
    --lr_decay_rate=0.96 \
    --batch_size=64 \
    --num_sample_softmax=10 \
    --inference_log_step=2 \
    --save_step=5