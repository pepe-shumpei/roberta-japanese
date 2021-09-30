#!/bin/sh                                                                                                                                                  

PEAK_LR=0.0005          # Peak learning rate, adjust as needed
MAX_SENTENCES=25        # Number of sequences per batch (batch size)
UPDATE_FREQ=2          # Increase the batch size 16x

DATA_DIR=data-bin/wiki-ja

MKL_THREADING_LAYER=GNU \
CUDA_VISIBLE_DEVICES=0,1 \
fairseq-train $DATA_DIR --arch roberta_base --task masked_lm --criterion masked_lm \
    --tokens-per-sample 512 --max-sentences $MAX_SENTENCES \
    --max-update 500000 --total-num-update 100000 --update-freq $UPDATE_FREQ \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates 10000 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
    --no-epoch-checkpoints --seed 88 --log-format simple --log-interval 1000 --save-interval-updates 5000 \
    --fp16 --num-workers 0 --fp16-init-scale 4 2>&1 | tee train.log

