python3 -m hred.train \
    --train_path=/home/shuuki4/sandbox_project/HRED/toy_data/train.txt \
    --val_path=/home/shuuki4/sandbox_project/HRED/toy_data/val.txt \
    --vocab_path=/home/shuuki4/sandbox_project/HRED/toy_data/vocab.txt \
    --save_dir=/home/shuuki4/sandbox_project/HRED/toy_data/$1 \
    --word_embedding_dim=128 \
    --num_rnn_layers=2 \
    --num_rnn_units=256
