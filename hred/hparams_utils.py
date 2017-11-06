import tensorflow as tf
import argparse


def get_hparams_parser():
    parser = argparse.ArgumentParser()

    # dataset configs
    parser.add_argument("--train_path", type=str, default=None,
                        help="path of training corpus")
    parser.add_argument("--val_path", type=str, default=None,
                        help="path of validation corpus")
    parser.add_argument("--vocab_path", type=str, default=None,
                        help="path of vocabulary file")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="directory for saving logs/checkpoints")
    parser.add_argument("--load_checkpoint_path", type=str, default='',
                        help="if given, load from checkpoint")

    # word configs
    parser.add_argument("--word_embedding_dim", type=int, default=256,
                        help="dimension of word embeddings")
    parser.add_argument("--pretrained_word_path", type=str, default=None,
                        help="path of pretrained word embeddings")
    parser.add_argument("--num_sample_softmax", type=int, default=2048,
                        help="number of samples to sample for sampled softmax")

    # rnn configs
    parser.add_argument("--rnn_cell_type", type=str, default="lstm",
                        help="cell type of rnn, lstm|gru|layernormlstm")
    parser.add_argument("--num_rnn_layers", type=int, default=4,
                        help="number of stacked layers in rnn")
    parser.add_argument("--num_bidi_layers", type=int, default=1,
                        help="number of bidirectional layers in encoder")
    parser.add_argument("--num_rnn_units", type=int, default=512,
                        help="number of rnn units")
    parser.add_argument("--dropout_keep_prob", type=float, default=0.7,
                        help="keep probability of dropouts")
    parser.add_argument("--attention_type", type=str, default="luong",
                        help="attention architecture. bahdanau|luong")
    parser.add_argument("--scheduled_sampling_decay_steps",
                        type=int, default=10000,
                        help="decay steps for scheduled sampling")
    parser.add_argument("--scheduled_sampling_decay_rate",
                        type=float, default=0.96,
                        help="decay ratio for scheduled sampling")

    # optimizer configs
    parser.add_argument("--optimizer", type=str, default="adam",
                        help="type of optimizer to use")
    parser.add_argument("--initial_learning_rate", type=float, default=1e-3,
                        help="initial value of learning rate")
    parser.add_argument("--start_decay_step", type=int, default=10000,
                        help="step to start learning rate decay")
    parser.add_argument("--lr_decay_steps", type=int, default=10000,
                        help="decay steps for learning rate decay")
    parser.add_argument("--lr_decay_rate", type=float, default=0.96,
                        help="decay rate for learning rate decay")
    parser.add_argument("--min_decay_rate", type=float, default=0.01,
                        help="minimum decay rate")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="momentum value for momentum optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=5.0,
                        help="maximum value of global norm")

    # training configs
    parser.add_argument("--batch_size", type=int, default=64,
                        help="size of single mini-batch")
    parser.add_argument("--max_epoch", type=float, default=100,
                        help="number of epoches to run")
    parser.add_argument("--loss_log_step", type=int, default=50,
                        help="frequency of steps to log loss")
    parser.add_argument("--inference_log_step", type=int, default=200,
                        help="frequency of steps to log train inference")
    parser.add_argument("--summary_step", type=int, default=20,
                        help="frequency of steps to save summary stats")
    parser.add_argument("--save_step", type=int, default=10000,
                        help="frequency of steps to save checkpoints")

    # validation configs
    parser.add_argument("--val_batch_size", type=int, default=64,
                        help="size of single mini-batch for validation")
    parser.add_argument("--val_step", type=int, default=10000,
                        help="frequency of steps for validation")

    # inference configs
    parser.add_argument("--beam_width", type=int, default=10,
                        help="width of beam used in beam search")
    return parser
