import re
import json
import string
import spacy
import argparse
import collections
from spacy.lang.en import English
from utils import read_fin10K

parser = argparse.ArgumentParser()
parser.add_argument("-input_path", "--path_input_file", type=str)
parser.add_argument("-output_path", "--path_output_file", type=str)
parser.add_argument("-model_type", "--model_type", type=str)
args = parser.parse_args()

nlp = English()

def token_extraction(srcA, srcB):
    """ Convert the sentence pairs into the extracted tokens, which are ready to predict.

    [TODO] position-sensitive keyword
    [TODO] integrate this function for data preprocessing codes with (for training)
    [CONCERN] for now, i only truncate ",.!?"
    """
    tokens_A, tokens_B = list(), list()

    for tok in nlp(srcA):
        tokens_A += [tok.text]
    for tok in nlp(srcB):
        tokens_B += [tok.text]

    return {'sentA': ' '.join(tokens_A),
            'sentB': ' '.join(tokens_B),
            'wordsA': tokens_A,
            'wordsB': tokens_B,
            'keywordsA': [], 
            'keywordsB': [],
            'labels': []}

def convert_to_bert(args):
    """ Only made for 'bert-seq-labeling' """

    data = read_fin10K(args.path_input_file)

    with open(args.path_output_file, 'w') as f:
        for i, (sa, sb) in enumerate(zip(data['sentA'], data['sentB'])):
            example = token_extraction(sa, sb)
            f.write(json.dumps(example) + '\n')

def convert_to_t5(args):
    """ Only made for 't5-marks-generation' """
    data = read_fin10K(args.path_input_file)

    with open(args.path_output_file, 'w') as f:
        for i, (sa, sb) in enumerate(zip(data['sentA'], data['sentB'])):
            f.write(f'Sentence1: {sa} Sentence2: {sb} Highlight:\n')

if args.model_type == 'bert':
    convert_to_bert(args)
elif args.model_type == 't5':
    convert_to_t5(args)
print("Done")
