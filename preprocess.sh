#!/bin/sh                                                                                                                                                  

DIR=data/wiki/tokenized

fairseq-preprocess \
    --only-source \
    --trainpref $DIR/wiki.train \
    --validpref $DIR/wiki.valid \
    --testpref $DIR/wiki.test \
    --destdir data-bin/wiki-ja \
    --workers 60
