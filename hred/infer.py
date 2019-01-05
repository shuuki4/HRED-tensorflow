import tensorflow as tf
from .hparams_utils import get_hparams_parser
import os
from .utils import log
from .model_helper import get_infer_model
from .dataset import to_single_string

FLAGS = None


def infer(argv=None):
    hparams = tf.contrib.training.HParams(**vars(FLAGS))
    assert os.path.exists(hparams.save_dir)

    with open(os.path.join(hparams.save_dir, 'hparams.json'),
              'r', encoding='utf-8') as f:
        hparams_json_string = f.read()
        hparams.parse_json(hparams_json_string)

    # gpu is not needed in inference
    # tf_config = tf.ConfigProto()
    # tf_config.gpu_options.allow_growth = True

    infer_graph = tf.Graph()
    with infer_graph.as_default():
        context_queries = hparams.infer_sample.split('\t')
        # num_sentence is the placeholder for size of input_strings
        infer_model, infer_inputs, infer_sess = get_infer_model(
            hparams, infer_graph, num_sentence=len(context_queries))

        # infer_sess.run(tf.global_variables_initializer())
        hparams.load_checkpoint_path = hparams.save_dir

        if not hparams.load_checkpoint_path:
            raise ValueError('no checkpoint path to load')
        log.info('restoring latest checkpoint from {}'.format(hparams.load_checkpoint_path))
        infer_model.saver.restore(infer_sess, tf.train.latest_checkpoint(hparams.load_checkpoint_path))

    log.info("infering for queries chain: {}".format(' -> '.join(context_queries)))

    res = infer_model.inference(infer_sess, feed_dict={infer_inputs.placeholder: context_queries})
    log.info("inferred following query: {}".format(to_single_string(res.inferences[0, :, 0])))

    log.info('Inference Done!')


if __name__ == '__main__':
    parser = get_hparams_parser()
    parser.add_argument("--num_vocab", type=int, default=0, help="will be set automatically, added for params parsing "
                                                                 "while restoring")
    FLAGS, unknown = parser.parse_known_args()
    tf.app.run(infer)
