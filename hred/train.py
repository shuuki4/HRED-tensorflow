import tensorflow as tf
import os

from .dataset import get_vocab_size, to_single_string
from .hparams_utils import get_hparams_parser
from .model_helper import get_model
from .utils import log


FLAGS = None


def validate(iterator, model, sess):
    # TODO: implement validate
    pass


def print_inference(source_strings, target_strings,
                    inference_strings, to_show=5):
    for i, (s_s, t_s, i_s) in enumerate(zip(source_strings[:to_show],
                                            target_strings[:to_show],
                                            inference_strings[:to_show])):
        for j, (_s_s, _t_s, _i_s) in enumerate(zip(s_s, t_s, i_s)):
            log.info('[Example {}, Time {}] {} -> {} (Real: {})'.format(
                i+1, j+1,
                to_single_string(_s_s),
                to_single_string(_i_s),
                to_single_string(_t_s)))


def train(argv=None):
    num_vocab = get_vocab_size(FLAGS.vocab_path)

    # TODO: add hparams management. save epoch in hparams.
    hparams = tf.contrib.training.HParams(**vars(FLAGS))
    hparams.add_hparam('num_vocab', num_vocab)

    if not os.path.exists(hparams.save_dir):
        os.makedirs(hparams.save_dir)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    global_step = 0
    train_graph = tf.Graph()
    with train_graph.as_default():
        train_model, train_iterator, train_sess = get_model(
            hparams, tf.contrib.learn.ModeKeys.TRAIN, train_graph)
        if hparams.load_checkpoint_path:
            train_model.saver.restore(train_sess, hparams.load_checkpoint_path)
            global_step = train_model.global_step.eval(train_sess)
        else:
            train_sess.run(tf.global_variables_initializer())

    val_graph = tf.Graph()
    with val_graph.as_default():
        val_model, val_iterator, val_sess = get_model(
            hparams, tf.contrib.learn.ModeKeys.EVAL, val_graph)

    summary_writer = tf.summary.FileWriter(FLAGS.save_dir, train_graph)

    epoch = 1
    train_sess.run(train_iterator.initializer)

    while epoch <= hparams.max_epoch:
        if global_step % hparams.save_step == 0:
            if global_step > 0:
                train_model.saver.save(train_sess,
                                       hparams.save_dir,
                                       global_step=global_step)
        if global_step % hparams.val_step == 0:
            latest_checkpoint = tf.train.latest_checkpoint(hparams.save_dir)
            if latest_checkpoint:
                val_model.saver.restore(val_sess, latest_checkpoint)
                validate(val_iterator, val_model, val_sess)
                # save to summary...
        try:
            step_result = train_model.train(
                train_sess,
                log_loss=global_step % hparams.loss_log_step == 0,
                log_inference=global_step % hparams.inference_log_step == 0,
                summary=global_step % hparams.summary_step == 0)

            if 'loss' in step_result:
                log.info('[Epoch {}] Step: {}, Loss: {:.5f}'
                         .format(epoch, global_step, step_result.loss))
            if 'inferences' in step_result:
                print_inference(
                    step_result.sources,
                    step_result.targets,
                    step_result.inferences,
                    to_show=5)
            if 'summary' in step_result:
                summary_writer.add_summary(step_result.summary, global_step)
                summary_writer.flush()

            global_step += 1

        except tf.errors.OutOfRangeError:  # epoch done
            log.warning('[Epoch {}] Epoch done!'.format(epoch))
            latest_checkpoint = tf.train.latest_checkpoint(hparams.save_dir)
            if latest_checkpoint:
                val_model.saver.restore(
                    val_sess, tf.train.latest_checkpoint(hparams.save_dir))
                validate(val_iterator, val_model, val_sess)
                # save to summary...

            epoch += 1
            train_sess.run(train_iterator.initializer)

    train_model.saver.save(train_sess, hparams.save_dir,
                           global_step=global_step)
    log.warning('Training Done!')


if __name__ == '__main__':
    parser = get_hparams_parser()
    FLAGS, unknown = parser.parse_known_args()
    tf.app.run(train)
