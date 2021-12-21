import os
import re
import string
import argparse
import collections
from utils import read_esnli

parser = argparse.ArgumentParser()
parser.add_argument("-data_dir", "--path_data_dir", type=str)
parser.add_argument("-output_path", "--path_output_file", type=str)
parser.add_argument("-target", "--target_type", type=str)
parser.add_argument("-class", "--class_selected", default="all", type=str)
parser.add_argument("--reverse", action="store_true", default=False)
args = parser.parse_args()

def extract_marked_token(sentence): 
    sentence = sentence.strip()
    token_list = []
    p_highlight = re.compile(r"[\*].*?[\*]")
    p_punct = re.compile("[" + re.escape(string.punctuation) + "]")
    findings = p_highlight.findall(sentence)

    for token in findings:
        token = p_punct.sub("", token)
        token_list += [token]

    if len(token_list) == 0:
        return "."
    else:
        return " ||| ".join(token_list)

def conditional_prefix(srcA, srcB, tgtA, tgtB, classes):

    if classes == 'contradiction':
        return f"Sentence1: {srcA} Sentence2: {srcB} Contradiction:\t{tgtB}\n"
    elif classes == 'entailment':
        return f"Sentence1: {srcA} Sentence2: {srcB} Entailment:\t{tgtB}\n"
    else:
        return f"Sentence1: {srcA} Sentence2: {srcB} Neutral:\t{tgtB}\n"

def convert_to_marks_generation(args):
    """
    a) Original NLI classification task
    b) Highlight 
        - class contraction only
    b) Conditional highlight
    """

    data = read_esnli(args)

    output = open(args.path_output_file, 'w')
    for sa, sb, hla, hlb, lbl in zip(data['sentA'], data['sentB'], data['highlightA'], data['highlightB'], data['label']):

        if args.target_type == "classification":
            example = f"Hypothesis: {sa} Premise: {sb} Relation:\t{lbl}\n"

        if args.target_type == "highlight":
            example = f"Sentence1: {sa} Sentence2: {sb} Highlight:\t{hlb}\n"

        if args.target_type == "highlight_conditional":
            example = conditional_prefix(sa, sb, hla, hlb, lbl)
        
        if args.target_type == "highlight_extraction":
            hlb_toks = extract_marked_token(hlb)
            example = f"Sentence1: {sa} Sentence2: {sb} Highlight:\t{hlb_toks}\n"

        output.write(example)

convert_to_marks_generation(args)
print("DONE")
