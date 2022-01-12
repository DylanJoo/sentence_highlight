import re
import json
import string
import argparse
import collections

parser = argparse.ArgumentParser()
parser.add_argument("-file", "--path_file", type=str)
args = parser.parse_args()

with open(args.path_file, 'r') as f:

    for line in f:
        pair = json.loads(line.strip())

        sosb = pair['word'].index('<tag2>')
        words = list(zip(pair['word'], pair['prob']))
        words.sort(key=lambda x: x[1])

        print(f"\nSentenceA: {' '.join(pair['word'][:sosb])}")
        print(f"SentenceB: {' '.join(pair['word'][(sosb+1): ])}")
        print(f"TOP 5: {words[::-1][:5]}\n")
