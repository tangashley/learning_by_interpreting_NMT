#!/usr/bin/env bash
SCRIPTS=mosesdecoder/scripts
GPU=0

SRC=fr
TRG=en
TEXT=iwslt17.tokenized.fr-en

#(7) Get attention information for train data
DATA_BIN=data-bin/iwslt17_${SRC}_${TRG}
CPKT=checkpoint/iwslt17_xfmr_${SRC}_${TRG}
ATTN_PATH=iwslt17_xfmr_saved_attn_${SRC}_${TRG}

mkdir -p $ATTN_PATH

python fairseq/fairseq_cli/interactive.py $DATA_BIN \
    --path $CPKT/checkpoint_best.pt \
    --buffer-size 2000 --batch-size 128 \
    --beam 5 --remove-bpe \
    --print-alignment \
    --save-attn \
    --attn-save-dir $ATTN_PATH/train/ \
    --input $TEXT/train.bpe.${SRC}

echo "Done with train data."

python fairseq/fairseq_cli/interactive.py $DATA_BIN \
    --path $CPKT/checkpoint_best.pt \
    --buffer-size 2000 --batch-size 128 \
    --beam 5 --remove-bpe \
    --print-alignment \
    --save-attn \
    --attn-save-dir $ATTN_PATH/valid/ \
    --input $TEXT/valid.bpe.${SRC}

echo "Done with validation data."

python fairseq/fairseq_cli/interactive.py $DATA_BIN \
    --path $CPKT/checkpoint_best.pt \
    --buffer-size 2000 --batch-size 128 \
    --beam 5 --remove-bpe \
    --print-alignment \
    --save-attn \
    --attn-save-dir $ATTN_PATH/test/ \
    --input $TEXT/test.bpe.${SRC}

echo "Done with test data."
