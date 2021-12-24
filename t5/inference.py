"""
This program is made for google colab's TPU setting, so make sure the following requirments are ready.

- TPU 2-8 in Google colabs
- GCP storage (buckets)
    - GCP authentication
"""
import functools
import os
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import t5
import t5.models as t5models
import seqio
import tensorflow_gcs_config
from google.colab import auth
from contextlib import contextmanager
import logging as py_logging
import argparse
import t5.data.mixtures

def main(args):
    """
    (0) TPU preparation 
    (1) Prepare model cofiguration
    (2) Predict
    (3) Download the output files
    """
    MODEL_SIZE = args.model_size if args.model_size else 'base'
    # Public GCS path for T5 pre-trained model checkpoints
    # FINETUNE_STEPS = args.train_steps
    # PRETRAINED_DIR = os.path.join("gs://t5-data/pretrained_models", MODEL_SIZE)
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    MODEL_DIR = os.path.join(MODELS_DIR, MODEL_SIZE)
    MODEL_DIR = os.path.join(MODEL_DIR, args.task_type)

    # Set parallelism and batch size to fit on v2-8 TPU (if possible).
    # Limit number of checkpoints to fit within 5GB (if possible).
    model_parallelism, train_batch_size, keep_checkpoint_max = {
        "small": (1, 256, 16),
        "base": (2, 128, 8),
        "large": (8, 64, 4),
        "3B": (8, 16, 1),
        "11B": (8, 16, 1)}[MODEL_SIZE]

    tf.io.gfile.makedirs(MODEL_DIR)
    # The models from our paper are based on the Mesh Tensorflow Transformer.
    model = t5models.MtfModel(
        model_dir=MODEL_DIR,
        tpu=TPU_ADDRESS,
        tpu_topology=TPU_TOPOLOGY,
        model_parallelism=model_parallelism,
        batch_size=16 if args.eval_batch_size is None else args.eval_batch_size,
    )

    model.predict(
        input_file=args.input_file_gs,
        output_file=args.input_file_gs, # the file will append the checkpoint steps by default
        checkpoint_steps=args.infer_steps,
        beam_size=args.beam_size,
        temperature=0
    )

    os.system(f'gsutil cp {args.output_file_gs}* {args.output_dir}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_gs", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--task_type", type=str, default="marks-generation")
    parser.add_argument("--model_size", type=str, default="base")
    parser.add_argument("--infer_steps", type=int, default=-1)
    parser.add_argument("--beam_size", type=int, default=1)
    args = parser.parse_args()

    # ***** Google's TPU preparation ******
    BASE_DIR = args.base_dir
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    print("Setting up GCS access...")
    # Set credentials for GCS reading/writing from Colab and TPU.
    TPU_TOPOLOGY = "v2-8"
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
        TPU_ADDRESS = tpu.get_master()
        print('Running on TPU:', TPU_ADDRESS)
    except ValueError:
        raise BaseException('ERROR: Not connected to a TPU runtime; please see the previous cell in this notebook for instructions!')

    auth.authenticate_user()
    tf.enable_eager_execution()
    tf.config.experimental_connect_to_host(TPU_ADDRESS)
    tensorflow_gcs_config.configure_gcs_from_colab_auth()
    tf.disable_v2_behavior()
    tf.get_logger().propagate = False
    py_logging.root.setLevel('INFO')

    @contextmanager
    def tf_verbosity_level(level):
        og_level = tf.logging.get_verbosity()
        tf.logging.set_verbosity(level)
        yield
        tf.logging.set_verbosity(og_level)

    main(args)
