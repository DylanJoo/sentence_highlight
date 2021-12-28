import os
import json
import collections

def read_fin10K(path):
    """ Function for reading the sentence A/B from parsed financial 10K report."""
    data = collections.defaultdict(list)

    with open(path, 'r') as f:
        for i, line in enumerate(f):
            partA, partB, scores = line.strip().split('|-|')
            idA, sentA = partA.strip().split('\t')
            idB, sentB = partB.strip().split('\t')
            score_sparse, score_dense = list(map(float, scores.strip().split()))

            data['idA'].append(idA)
            data['sentA'].append(sentA)
            data['idB'].append(idB)
            data['sentB'].append(sentB)
            data['scores'].append( (score_sparse, score_dense) )
    return data

def read_esnli(data_dir, split, class_selected, reverse):
    """ Function for reading the sentence A/B and highlight A/B with the corresponding labels """
    data = collections.defaultdict(list)

    with open(os.path.join(data_dir, f'esnli-{split}.jsonl'), 'r') as f:
        for i, item_dict in enumerate(f):
            items = json.loads(item_dict.strip())
            data['sentA'].append(items['Sentence1'])
            data['sentB'].append(items['Sentence2'])
            data['highlightA'].append(items['Marked1'])
            data['highlightB'].append(items['Marked2'])
            data['label'].append(items['label'])

    # example filtering 
    if class_selected != 'all':
        data['sentA'] = [h for (h, l) in zip(data['sentA'], data['label']) if l in class_selected]
        data['sentB'] = [h for (h, l) in zip(data['sentB'], data['label']) if l in class_selected]
        data['highlightA'] = [h for (h, l) in zip(data['highlightA'], data['label']) if l in class_selected]
        data['highlightB'] = [h for (h, l) in zip(data['highlightB'], data['label']) if l in class_selected]
        data['label'] = [l for l in data['label'] if l in class_selected]

    if reverse:
        data['sentA'] = data['sentA'] + data['sentB']
        data['sentB'] = data['sentB'] + data['sentA']
        data['highlightA'] = data['highligthA'] + data['hightlightB']
        data['highlightB'] = data['highligthB'] + data['hightlightA']
        data['label'] = data['label'] + data['label']

    return data

