#!/usr/bin/env bash

INPATH=iwslt17.tokenized.fr-en
OUTPATH=iwslt17.tokenized_xfmr_local_merged_words.fr-en
SRC=fr
TRG=en
ATTN_PATH=iwslt17_xfmr_local_saved_attn_${SRC}_${TRG}


# Concatenate words with the most attention with the source sequence.
python utils/merge_attn_with_source.py \
  --source $SRC \
  --target $TRG \
  --attn_path $ATTN_PATH \
  --inpath $INPATH \
  --outpath $OUTPATH \
  --merge_attn_words

TEXT=iwslt17.tokenized.fr-en

# copy target files to OUTPATH folder to simplify later process
cp $TEXT/*$TRG $OUTPATH/


SCRIPTS=mosesdecoder/scripts
GPU=0
LANG=fr-en

# train/valid data folder
TEXT=$OUTPATH

# folder for original binary train/valid data
ORG_DATA_BIN=data-bin/iwslt17_${SRC}_${TRG}

# folder for the binary files
DATA_BIN=data-bin/iwslt17_xfmr_local_merged_words_${SRC}_${TRG}

mkdir -p $DATA_BIN

# (5) Word to Integer Sequence (for concatenated input)
python fairseq/fairseq_cli/preprocess.py \
    --source-lang ${SRC} --target-lang ${TRG} \
    --trainpref $TEXT/train.bpe --validpref $TEXT/valid.bpe \
    --destdir $DATA_BIN \
    --tgtdict $ORG_DATA_BIN/dict.${TRG}.txt \
    --srcdict $ORG_DATA_BIN/dict.${SRC}.txt \
    --workers 20
    
CPKT=checkpoint/iwslt17_xfmr_local_merged_words_${SRC}_${TRG}
LOG=log/iwslt17_xfmr_local_merged_words_${SRC}_${TRG}

# mkdir -p  $CPKT $LOG

ORG_MODEL=checkpoint/iwslt17_xfmr_local_${SRC}_${TRG}/checkpoint_best.pt

#(6) Fine-tune NMT Model (Transformer)

CUDA_VISIBLE_DEVICES=$GPU python fairseq/fairseq_cli/train.py $DATA_BIN \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --finetune-from-model $ORG_MODEL \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.1 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --max-epoch 50 --patience 5 \
    --save-dir $CPKT | tee -a $LOG/train_transformer.out 
