import os
import re
import string
import argparse
import collections
from utils import read_esnli

parser = argparse.ArgumentParser()
parser.add_argument("-data_dir", "--path_data_dir", type=str)
parser.add_argument("-output_path", "--path_output_file", type=str)
parser.add_argument("-split", "--split", type=str)
parser.add_argument("-task", "--task_type", type=str)
parser.add_argument("-class", "--class_selected", default="all", type=str)
parser.add_argument("--reverse", action="store_true", default=False)
args = parser.parse_args()

# [TODO] Prepare the predcition-specific type of codes.
def extract_marked_token(sentence): 
    sentence = sentence.strip() if sentence is not False else ""
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
        return f"Sentence1: {srcA} Sentence2: {srcB} Contradiction:\t{tgtB}"
    elif classes == 'entailment':
        return f"Sentence1: {srcA} Sentence2: {srcB} Entailment:\t{tgtB}"
    else:
        return f"Sentence1: {srcA} Sentence2: {srcB} Neutral:\t{tgtB}"

def convert_to_marks_generation(args):
    data = read_esnli(args.path_data_dir, args.split, args.class_selected, args.reverse)

    output = open(args.path_output_file, 'w')
    for sa, sb, hla, hlb, lbl in zip(data['sentA'], data['sentB'], 
                                     data['highlightA'], data['highlightB'], data['label']):

        if args.task_type == "classification":
            example = f"Hypothesis: {sa} Premise: {sb} Relation:\t{lbl}"
        if args.task_type == "marks-generation":
            example = f"Sentence1: {sa} Sentence2: {sb} Highlight:\t{hlb}"
        if args.task_type == "marks-generation-conditional":
            example = conditional_prefix(sa, sb, hla, hlb, lbl)
        if args.task_type == "token-extraction":
            hlb_toks = extract_marked_token(hlb)
            example = f"Sentence1: {sa} Sentence2: {sb} Highlight:\t{hlb_toks}"

        if args.split != 'train':
            example = example.split('\t')[0]
        if hla and hlb:
            output.write(example + '\n')

convert_to_marks_generation(args)
print("Done")
