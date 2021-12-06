import re
import json
import string
import spacy
import argparse
import collections
from spacy.lang.en import English
from utils import read_esnli

parser = argparse.ArgumentParser()
parser.add_argument("-data_dir", "--path_data_dir", type=str)
parser.add_argument("-output_path", "--path_output_file", type=str)
parser.add_argument("-highlightB_only", "--labeling_on_sentB_only", action='store_true', default=False)
parser.add_argument("-class", "--class_selected", type=str, default='contradiction')
args = parser.parse_args()

nlp = English()

def keyword_extraction(srcA, srcB, tgtA, tgtB, only_on_B=True):
    """ Convert the "starred marks" to the separated list of elements. 
    Noted that, the default highlightA is False, only extract highlight B for simplicity.

    [TODO] position-sensitive keyword
    [CONCERN] for now, i only truncate ",.!?"
    """
    tgtA = srcA if tgtA is None else tgtA
    tgtB = srcB if tgtB is None else tgtB

    def highlight_process(sentence):
        tokens = list()
        tokens_hl = list()
        labels = list()
        hl = 0
        punc = (lambda x: x in [",", ".", "?", "!"]) 

        for tok in nlp(sentence):
            if tok.text == "*":
                hl = 0 if hl else 1
            else:
                if punc(tok.text): 
                    tokens_hl += []
                    labels += [0]
                else:
                    tokens_hl += [tok.text] if hl else []
                    labels += [1] if hl else [0]
                tokens += [tok.text]

        return tokens, tokens_hl, labels

    tokens_A, tokens_A_hl, labels_A = highlight_process(tgtA)
    tokens_B, tokens_B_hl, labels_B = highlight_process(tgtB)

    if only_on_B:
        labels_A = [-100] * len(labels_A)
        tokens_A_hl = []

    return {'sentA': ' '.join(tokens_A), 
            'sentB': ' '.join(tokens_B), 
            'keywordsA': tokens_A_hl, 
            'keywordsB': tokens_B_hl,
            'labels': labels_A + labels_B}

def convert_to_tokens_labeling(args):

    # read 5 parsed text esnli files
    data = read_esnli(args)

    with open(args.path_output_file, 'w') as f:
        for sa, sb, hla, hlb in zip(data['sentA'], data['sentB'], data['highlightA'], data['highlightB']):
            example = keyword_extraction(sa, sb, hla, hlb, args.labeling_on_sentB_only)
            f.write(json.dumps(example) + '\n')

convert_to_tokens_labeling(args)
print('DONE')
