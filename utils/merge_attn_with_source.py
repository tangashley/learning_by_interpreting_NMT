import argparse
import os
from pathlib import Path
from shutil import copyfile as cp

from PIL.features import codecs

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--source', '-s', type=str, help='Source language')
parser.add_argument('--target', '-t', type=str, help='Target language')
parser.add_argument('--attn_path', '-a', type=str, default=None, help='Path to the file with attention information')
parser.add_argument('--merge_attn_pos', '-p', action='store_true', help='Merge positions with most attention')
parser.add_argument('--merge_attn_words', '-w', action='store_true', help='Merge words with most attention')
parser.add_argument('--inpath', '-i', type=str, help='Input path')
parser.add_argument('--filenames', '-f', type=str, default='train,valid,test', help='Input filenames comma separated')
parser.add_argument('--outpath', '-o', type=str, help='Output path')
args = parser.parse_args()

src = args.source
tgt = args.target
attn_path = args.attn_path
merge_attn_pos = args.merge_attn_pos
merge_attn_words = args.merge_attn_words
inpath = args.inpath
filenames = args.filenames.split(',')
outpath = args.outpath


for filename in filenames:
    print(src + " : " + filename)
    if not Path(outpath).exists():
        os.makedirs(outpath)
        print('Create directory: ' + outpath)

    if merge_attn_pos and merge_attn_words:
        with open(inpath + '/' + filename + '.bpe.' + src, 'r') as input_file:
            with open(attn_path + '/' + filename + '/' + 'attn_pos.txt', 'r') as attn_pos_file:
                with open(attn_path + '/' + filename + '/' + 'attn_words.txt', 'r') as attn_words_file:
                    with open(outpath + '/' + filename + '.bpe.' + src, 'w', encoding='utf-8') as output_file:
                        for src_line, pos_line, word_line in zip(input_file, attn_pos_file, attn_words_file):
                            src_line = src_line.strip()
                            pos_line = pos_line.strip()
                            word_line = word_line.strip()
                            try:
                                output_file.write(src_line + ' ' + word_line + ' ' + pos_line + '\n')
                            except:
                                print("Line ", src_line)
                                exit()
    elif merge_attn_pos:
        with open(inpath + '/' + filename + '.bpe.' + src, 'r') as input_file:
            with open(attn_path + '/' + filename + '/' + 'attn_pos.txt', 'r') as attn_pos_file:
                with open(outpath + '/' + filename + '.bpe.' + src, 'w', encoding='utf-8') as output_file:
                    for src_line, pos_line in zip(input_file, attn_pos_file):
                        src_line = src_line.strip()
                        pos_line = pos_line.strip()
                        try:
                            output_file.write(src_line + ' ' + pos_line + '\n')
                        except:
                            print("Line ", src_line)
                            exit()
    elif merge_attn_words:
        with open(inpath + '/' + filename + '.bpe.' + src, 'r') as input_file:
            with open(attn_path + '/' + filename + '/' + 'attn_words.txt', 'r') as attn_words_file:
                with open(outpath + '/' + filename + '.bpe.' + src, 'w', encoding='utf-8') as output_file:
                    for src_line, word_line in zip(input_file, attn_words_file):
                        src_line = src_line.strip()
                        word_line = word_line.strip()
                        try:
                            output_file.write(src_line + ' ' + word_line + '\n')
                        except:
                            print("Line ", src_line)
                            exit()
print("DONE!")