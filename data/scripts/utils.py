import os
import collections


def readlines(filename):
    f = open(filename, 'r').readlines()
    data = list(map(lambda x: x.strip(), f))
    return data

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

def read_esnli(args):
    """ Function for reading the sentence A/B and highlight A/B with the corresponding labels """

    data = collections.OrderedDict()
    data['sentA'] = readlines(os.path.join(args.path_data_dir, 'sentenceA.txt'))
    data['sentB'] = readlines(os.path.join(args.path_data_dir, 'sentenceB.txt'))
    data['highlightA'] = readlines(os.path.join(args.path_data_dir, 'highlightA.txt'))
    data['highlightB'] = readlines(os.path.join(args.path_data_dir, 'highlightB.txt'))
    data['label'] = readlines(os.path.join(args.path_data_dir, 'label.txt'))

    if os.path.exists(os.path.join(args.path_data_dir, 'explanation.txt')):
        data['explanation'] = readlines(os.path.join(args.path_data_dir, 'explanation.txt'))

    # example filtering 
    if args.class_selected != 'all':
        data['sentA'] = [h for (h, l) in zip(data['sentA'], data['label']) if l in args.class_selected]
        data['sentB'] = [h for (h, l) in zip(data['sentB'], data['label']) if l in args.class_selected]
        data['highlightA'] = [h for (h, l) in zip(data['highlightA'], data['label']) if l in args.class_selected]
        data['highlightB'] = [h for (h, l) in zip(data['highlightB'], data['label']) if l in args.class_selected]
        data['label'] = [l for l in data['label'] if l in args.class_selected]

    if args.reverse:
        data['sentA'] = data['sentA'] + data['sentB']
        data['sentB'] = data['sentB'] + data['sentA']
        data['highlightA'] = data['highligthA'] + data['hightlightB']
        data['highlightB'] = data['highligthB'] + data['hightlightA']
        data['label'] = data['label'] + data['label']

    return data

