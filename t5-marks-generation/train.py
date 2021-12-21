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

def main(args):
    """
    (0) TPU preparation 
    (1) Data preparation
    """
    if args.task_type == 'marks-generation':
        TRAIN_FILE = os.path.joins(BASE_DIR, 'data/esnli_sents_highlight_contradict_pairs.tsv')
    elif args.task_type == 'token-extraction'
        TRAIN_FILE = os.path.joins(BASE_DIR, 'data/esnli_sents_highlight_extraction_pairs.tsv')

    # ***** 1a *****
    def esnli_highlight_ds(split, shuffle_files):
        '''The tfText dataset pipeline.  '''
        dataset = tf.data.TextLineDataset(TRAIN_FILE)
        dataset = dataset.map(
            functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                              field_delim="\t", use_quote_delim=False),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
                lambda *ex: dict(zip(["src", "tgt"], ex))
        )
        return dataset

    def esnli_highlight_prep(ds):
        '''The preprocessor of t5 source and target '''
        def normalize_text(text):
            # text = tf.strings.lower(text)
            # text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
            return text

        def to_inputs_and_targets(ex):
            return {
                "inputs": normalize_text(ex["src"]), 
                "targets": normalize_text(ex["tgt"])
            }
        return ds.map(to_inputs_and_targets,
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # ****** 2  *****
    t5.data.TaskRegistry.remove(args.task_type)
    t5.data.TaskRegistry.add(
            args.task_type
            dataset_fn=esnli_highlight_ds, 
            splits=["train"],
            text_preprocessor=[esnli_highlight_prep]
    )
    demo = t5.data.TaskRegistry.get(args.task_type)
    ds = train.get_dataset(split="train", sequence_length={"inputs": 512, "targets": 64}, shuffle=True)
    print("[ESNLI]: A few preprocessed training examples...")
    for ex in tfds.as_numpy(ds.take(1)):
        print(ex)

    # ***** 3  *****
    MDOEL_SIZE = arg.model_size if args.model_size else 'base'
    # Public GCS path for T5 pre-trained model checkpoints
    FINETUNE_STEPS = args.train_steps
    PRETRAINED_DIR = os.path.join("gs://t5-data/pretrained_models", MODEL_SIZE)
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
        sequence_length={"inputs": args.max_src_len, "targets": args.max_tgt_len},
        batch_size=train_batch_size if arg.train_batch_size is None else args.train_batch_size,
        # batch_size=("tokens_per_batch", 65536),
        learning_rate_schedule=0.001,
        save_checkpoints_steps=1000,
        keep_checkpoint_max=keep_checkpoint_max,
        iterations_per_loop=100,
    )

    model.finetune(
        mixture_or_task_name=args.task_type,
        finetune_steps=args.train_steps,
        pretrained_model_dir=PRETRAINED_DIR,
        pretrained_checkpoint_step=-1,
        split="train"
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, require=True)
    parser.add_argument("--task_type", type=str, default="marks-generation")
    parser.add_argument("--model_size", type=str, default="base")
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--train_steps", type=int, default=4000)
    parser.add_argument("--max_src_len", type=int, default=512)
    parser.add_argument("--max_tgt_len", type=int, default=64)
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
