import re
import json
import string
import spacy
import argparse
import collections
from spacy.lang.en import English

parser = argparse.ArgumentParser()
parser.add_argument("-out", "--path_output", type=str)
parser.add_argument("-sentA", "--path_sentenceA", type=str)
parser.add_argument("-sentB", "--path_sentenceB", type=str)
parser.add_argument("-highlightA", "--path_highlightA", type=str)
parser.add_argument("-highlightB", "--path_highlightB", type=str)
parser.add_argument("-label", "--path_labels", type=str)
parser.add_argument("-class", "--class_selected")
args = parser.parse_args()

nlp = English()

def keyword_extraction(srcA, srcB, tgtA, tgtB, highlightA=False):
    """ Convert the "starred marks" to the separated list of elements. 
    Noted that, the default highlightA is False, only extract highlight B for simplicity.

    [TODO] position-sensitive keyword
    [TODO] first tokenize with word-level, and extract the marked tokens.
    """
    tgtA = srcA if tgtA is None else tgtA
    tgtB = srcB if tgtB is None else tgtB

    def highlight_process(sentence):
        tokens = list()
        tokens_hl = list()
        labels = list()
        hl = 0
        punc = (lambda x: x in [",", ".", "?", "!"]) # [CONCERN] for now, i only truncate ",.!?"

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

    if highlightA is False:
        labels_A = [0] * len(labels_A)
        tokens_A_hl = []

    return {'sentA': ' '.join(tokens_A), 
            'sentB': ' '.join(tokens_B), 
            'keywordsA': tokens_A_hl, 
            'keywordsB': tokens_B_hl,
            'labels': labels_A + labels_B}

def create_highlight_list(args):

    def readlines(filename):
        f = open(filename, 'r').readlines()
        data = list(map(lambda x: x.strip(), f))
        return data

    data = collections.OrderedDict()
    data['sentenceA'] = readlines(args.path_sentenceA)
    data['sentenceB'] = readlines(args.path_sentenceB)
    data['highlightA'] = readlines(args.path_highlightA)
    data['highlightB'] = readlines(args.path_highlightB)
    data['label'] = readlines(args.path_labels)

    if args.class_selected != 'all':
        data['sentenceA'] = [h for (h, l) in zip(data['sentenceA'], data['label']) \
                if l in args.class_selected]
        data['sentenceB'] = [h for (h, l) in zip(data['sentenceB'], data['label']) \
                if l in args.class_selected]
        data['highlightA'] = [h for (h, l) in zip(data['highlightA'], data['label']) \
                if l in args.class_selected]
        data['highlighB'] = [h for (h, l) in zip(data['highlightB'], data['label']) \
                if l in args.class_selected]
        data['label'] = [l for  l in data['label'] if l in args.class_selected]

    with open(args.path_output, 'w') as f:
        for sa, sb, hla, hlb in zip(data['sentenceA'], data['sentenceB'], \
                                    data['highlightA'], data['highlightB']):
            example = keyword_extraction(srcA=sa, srcB=sb, tgtA=hla, tgtB=hlb)
            f.write(json.dumps(example) + '\n')

create_highlight_list(args)
print('DONE')
