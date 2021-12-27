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
args = parser.parse_args()

nlp = English()

def convert_to_sents_text_pairs(args):
    data = read_fin10K(args.path_input_file)

    with open(args.path_output_file, 'w') as f:
        for i, (sa, sb) in enumerate(zip(data['sentA'], data['sentB'])):
            f.write(f'Sentence1: {sa} Sentence2: {sb} Highlight:\n')

convert_to_sents_text_pairs(args)
print("DONE")
