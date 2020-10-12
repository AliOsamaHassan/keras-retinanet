from keras_retinanet import models
from tensorflow.python import saved_model
import shutil
import argparse
import os
import sys
import keras
import tensorflow as tf


# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"


def parse_args(args):
    parser = argparse.ArgumentParser(description='Script for uploading inference model to Google Cloud Storage.')
    parser.add_argument('model_in', help='The model to convert.')
    parser.add_argument('model_out', help='Path to save the converted model to.')
    parser.add_argument('--backbone', help='The backbone of the model to convert.', default='resnet50')
    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    model = models.load_model(args.model_in, backbone_name='resnet50')

    session = tf.compat.v1.keras.backend.get_session()
    tf.keras.backend.set_learning_phase(0) # Ignore dropout at inference

    with session as sess:
        tf.compat.v1.saved_model.simple_save(
                    session,
                    args.model_out,
                    inputs={'input_image': model.input},
                    outputs={t.name:t for t in model.outputs})


if __name__ == '__main__':
    main()