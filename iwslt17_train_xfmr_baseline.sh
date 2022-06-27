#!/usr/bin/env bash
SCRIPTS=mosesdecoder/scripts
GPU=0

SRC=fr
TRG=en
TEXT=iwslt17.tokenized.fr-en

DATA_BIN=data-bin/iwslt17_${SRC}_${TRG}
# mkdir -p $DATA_BIN

#(5) Word to Integer Sequence
# python fairseq/preprocess.py --source-lang ${SRC} --target-lang ${TRG} \
#     --trainpref $TEXT/train --validpref $TEXT/valid \
#     --destdir $DATA_BIN \
#     --workers 20
	
CPKT=checkpoint/iwslt17_xfmr_local_${SRC}_${TRG}
LOG=log/iwslt17_xfmr_local_${SRC}_${TRG}

mkdir -p  $CPKT $LOG

#(6) Train NMT Model (Transformer)
CUDA_VISIBLE_DEVICES=$GPU python fairseq/fairseq_cli/train.py $DATA_BIN \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.1 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --max-epoch 50 --patience 5 \
    --save-dir $CPKT | tee $LOG/train_transformer.out