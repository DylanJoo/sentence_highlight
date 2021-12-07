import collections
import argparse
import numpy as np
from spacy.lang.en import English
import json


parser = argparse.ArgumentParser()
parser.add_argument("-truth", "--path_truth_file", type=str)
parser.add_argument("-pred", "--path_pred_file", type=str)
parser.add_argument("-hl_type", "--output_type", type=str)
parser.add_argument("-eval_mode", "--evaluation_mode", type=str)
args = parser.parse_args()

nlp = English()

# truth jsonl file
def load_from_jsonl(file_path):
    truth = collections.OrderedDict()

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            truth[i] = json.loads(line)['keywordB']

    return truth

# prediction text files
def load_from_bert_lime(file_path, prob_threshold=0, topk=-1, topn=-1):
    """
    File type: dictionary file, e.g. .json, .jsonl
    """
    pass

def load_from_bert_seq_labeling(file_path, prob_threshold=0.5):
    """
    File type: dictionary file, e.g. .json, .jsonl
    (1) Post-process the token-to-word labeling.
    """
    pass

def load_from_bert_span_detection(file_path):
    pass

def load_from_t5_mark_generation(file_path):
    """
    File type: Raw text, e.g. .txt, .tsv
    """
    pred = collections.defaultdict(list)
    punc = (lambda x: x in [",", ".", "?", "!"])

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            hl = 0
            for tok in nlp(line.strip()):
                if tok.text == "*":
                    hl = 0 if hl else 1
                else:
                    if punc(tok.text):
                        pass
                    else:
                        pred[i] += [tok.text] if hl else []
    return pred


def main(args):
    truth = load_from_jsonl(args.path_truth_file)
    if args.output_type == 'bert_lime':
        pred = load_from_bert_lime(args.path_pred_file)
    elif args.output_type == 'bert_seq_labeling':
        pred = load_from_bert_seq_labeling(args.path_pred_file)
    elif args.output_type == 'bert_span_detection':
        pred = load_from_bert_span_detection(args.path_pred_file)
    elif args.output_type == 't5_marks_generation':
        pred = load_from_t5_mark_generation(args.path_pred_file)
    else:
        print("Invalid type of highlight tasks")
        exit(0)

    assert len(truth) != len(pred), "Inconsisent sizes of truth and predictions"
    metrics = collections.defaultdict(list)

    for i in range(len(truth)):
        # [CONCERN] what if the tokens are redundant, revised if needed.
        hits = len(set(truth[i]) & set(pred[i]))
        metrics['precision'].append( hits / len(pred[i]) )
        metrics['recall'].append( hits / len(truth[i]) )
        metrics['f1'].append( 
                2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
        )
    return "********************************\
            \nMean {:<9}: {:<5}\
            \nMean {:<9}: {:<5}\
            \nMean {:<9}: {:<5}\
            \nNum of evaluated samples: {}\
            \n********************************".format('precision', np.mean(metrics['precision']),
                                                       'recall', np.mean(metrics['recall']),
                                                       'f1-score', np.mean(metrics['f1']), len(truth))

if __name__ == "__main__()":
    main(args)




