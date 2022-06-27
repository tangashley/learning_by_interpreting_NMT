import random
import argparse
import os
from pathlib import Path
from shutil import copyfile as cp

from PIL.features import codecs

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--source', '-s', type=str, help='Source language')
parser.add_argument('--attn_path', '-a', type=str, default=None, help='Path to the file with attention information')
parser.add_argument('--inpath', '-i', type=str, help='Input path')
parser.add_argument('--filenames', '-f', type=str, default='train,valid,test', help='Input filenames comma separated')
parser.add_argument('--outpath', '-o', type=str, help='Output path')
parser.add_argument('--rand_prob', '-r', type=float, help='Probility of getting random words as attention words')
args = parser.parse_args()

src = args.source
attn_path = args.attn_path
inpath = args.inpath
filenames = args.filenames.split(',')
outpath = args.outpath
rand_prob = args.rand_prob

print("***************************")
print("random_prob is " + str(type(rand_prob)))
print("random_prob is " + str(rand_prob))
print("***************************")

for filename in filenames:
    print(src + " : " + filename)
    if not Path(outpath).exists():
        os.makedirs(outpath)
        print('Create directory: ' + outpath)

    with open(inpath + '/' + filename + '.bpe.' + src, 'r') as input_file:
        with open(attn_path + '/' + filename + '/' + 'attn_words.txt', 'r') as attn_words_file:
            with open(outpath + '/' + filename + '.bpe.' + src, 'w', encoding='utf-8') as output_file:
                for src_line, attn_word_line in zip(input_file, attn_words_file):
                    src_line = src_line.strip()
                    src_words = src_line.split()
                    attn_words = attn_word_line.strip().split()

                    num_rand = len(attn_words) * rand_prob
                    rand_idx = random.sample(range(len(attn_words)), int(num_rand))

                    attn_words_with_rand = []

                    for idx in range(len(attn_words)):
                        if idx not in rand_idx:
                            # get it from real attention word
                            attn_words_with_rand.append(attn_words[idx])
                        else:
                            # get it randomly from src words
                            rand_word_pos = random.randint(0, len(src_words) - 1)
                            count = 0
                            while src_words[rand_word_pos] == attn_words[idx]:
                                count += 1
                                rand_word_pos = random.randint(0, len(src_words) - 1)
                                if count > 1:
                                    # prevent endless loop if there's only one word in the src sentence.
                                    break

                            attn_words_with_rand.append(src_words[rand_word_pos])

                    try:
                        output_file.write(src_line + ' ' + ' '.join(attn_words_with_rand) + '\n')
                    except:
                        print("Line ", src_line)
                        exit()
print("DONE!")