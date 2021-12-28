import os
import re
import json
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-train1", "--path_esnli_train_1", default="esnli_train_1.csv")
parser.add_argument("-train2", "--path_esnli_train_2", default="esnli_train_2.csv")
parser.add_argument("-dev", "--path_esnli_dev", default="esnli_dev.csv")
parser.add_argument("-test", "--path_esnli_test", default="esnli_test.csv")
parser.add_argument('--split', action='append')
parser.add_argument("-output_dir", "--path_output_dir", default="preprocessed")
args = parser.parse_args()

def normalized(strings, mode=None):
    try:
        strings = strings.strip()
        strings = re.sub('"', '', strings)
        strings = re.sub(r"\t", "", strings)
        strings = re.sub(r"\n", " ", strings)
        strings = re.sub(r"\s+", " ", strings)
        return strings
    except:
        return False

def parse_esnli_csv(args, split):

    if split == 'train':
        data = pd.concat([pd.read_csv(args.path_esnli_train_1), 
                          pd.read_csv(args.path_esnli_train_2)], axis=0)
    elif split == 'dev':
        data = pd.read_csv(args.path_esnli_dev)
    elif split == 'test':
        data = pd.read_csv(args.path_esnli_test)

    data.reset_index(inplace=True)
    data = data.to_dict()
        
    # Extract the e-snli data
    with open(os.path.join(args.path_output_dir, f'esnli-{split}.jsonl'), 'w') as f:

        for index in data['index']:
            sentA = normalized(data['Sentence1'][index], 'sentA')
            sentB = normalized(data['Sentence2'][index], 'sentB')
            highA = normalized(data['Sentence1_marked_1'][index], 'sentA')
            highB = normalized(data['Sentence2_marked_1'][index], 'sentB')
            label = normalized(data['gold_label'][index], 'label')
           
            if sentA and sentB: 
                f.write(json.dumps({
                    'Sentence1': sentA, 'Sentence2': sentB,
                    'Marked1': highA, 'Marked2': highB, 
                    'label': label
                }) + '\n')

        print("Total number of data: {}".format(index))


if os.path.exists(args.path_output_dir) is False:
    os.system(f'mkdir {args.path_output_dir}')
# Read and preprocessed the file
for s in args.split:
    print(s)
    parse_esnli_csv(args, split=s)

print("Done")
