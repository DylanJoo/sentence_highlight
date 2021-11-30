import os
import re
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-train1", "--path_esnli_train_1", default="esnli_train_1.csv")
parser.add_argument("-train2", "--path_esnli_train_2", default="esnli_train_2.csv")
parser.add_argument("-dev", "--path_esnli_dev", default="esnli_dev.csv")
parser.add_argument("-test", "--path_esnli_test", default="esnli_test.csv")
parser.add_argument("-split", "--split", default="train")
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

    # train: includes two file
    # dev/test
    if split == 'train':
        data = pd.concat([pd.read_csv(args.path_esnli_train_1), pd.read_csv(args.path_esnli_train_2)], axis=0)
    elif args.split == 'dev':
        data = pd.read_csv(args.path_esnli_dev)
    elif args.split == 'test':
        data = pd.read_csv(args.path_esnli_test)

    data.reset_index(inplace=True)
    data = data.to_dict()
        
    # Extract the e-snli data
    with open(os.path.join(args.path_output_dir, f'{split}/sentenceA.txt'),'w') as s1, \
         open(os.path.join(args.path_output_dir, f'{split}/sentenceB.txt'),'w') as s2, \
         open(os.path.join(args.path_output_dir, f'{split}/highlightA.txt'),'w') as h1, \
         open(os.path.join(args.path_output_dir, f'{split}/highlightB.txt'), 'w') as h2, \
         open(os.path.join(args.path_output_dir, f'{split}/label.txt'), 'w') as lbl:

        for index in data['index']:
            sentA = normalized(data['Sentence1'][index], 'sentA')
            sentB = normalized(data['Sentence2'][index], 'sentB')
            highA = normalized(data['Sentence1_marked_1'][index], 'sentA')
            highB = normalized(data['Sentence2_marked_1'][index], 'sentB')
            label = normalized(data['gold_label'][index], 'label')
           
            if sentA and sentB: 
                s1.write(sentA + "\n")
                s2.write(sentB + "\n")
                if highA is False:
                    highA = sentA
                if highB is False:
                    highB = sentB

                h1.write(highA + "\n")
                h2.write(highB + "\n")
                lbl.write(label + "\n")

            if index % 100000 == 0:
                print("Preprocessing instance: {}".format(index))

        print("Total number of data: {}".format(index))

# Read and preprocessed the file
for s in args.split.split("|"):
    if os.path.exists(os.path.join(args.path_output_dir, s)) is False:
        os.system('mkdir {}'.format(args.path_output_dir))
        os.system('mkdir {}'.format(os.path.join(args.path_output_dir, s)))
        print("Directory Created")
    parse_esnli_csv(args, split=s)

print("DONE")
