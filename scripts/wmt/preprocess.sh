#!/bin/sh

# this sample script preprocesses a sample train, including tokenization,
# truecasing, and subword segmentation. 
# for application to a different language pair,
# change source and target prefix, optionally the number of BPE operations,
# and the file names (currently, data/train and data/devel are being processed)

# in the tokenization step, you will want to remove Romanian-specific normalization / diacritic removal,
# and you may want to add your own.
# also, you may want to learn BPE segmentations separately for each language,
# especially if they differ in their alphabet

# suffix of source language files
SRC=en

# suffix of target language files
TRG=it

# number of merge operations. Network vocabulary should be slightly larger (to include characters),
# or smaller if the operations are learned on the joint vocabulary
bpe_operations=40000

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=../

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=/path/to/subword-nmt 

# path to opennmt ( https://www.github.com/rsennrich/nematus )
opennmt=/path/to/opennmt

train=/train/corpus/name

devel=/development/corpus/name

# tokenize
for prefix in $train $devel
 do
   cat data/$prefix.$SRC | \
   $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $SRC | \
   $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $SRC > data/$prefix.tok.$SRC &

   cat data/$prefix.$TRG | \
   $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $TRG | \
   $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $TRG > data/$prefix.tok.$TRG &

 done
wait
# clean empty and long sentences, and sentences with high source-target ratio (training train only)
$mosesdecoder/scripts/training/clean-corpus-n.perl data/$train.tok $SRC $TRG data/$train.tok.clean 1 80

# train truecaser
$mosesdecoder/scripts/recaser/train-truecaser.perl --corpus data/$train.tok.clean.$SRC --model model/truecase-model.$SRC &
$mosesdecoder/scripts/recaser/train-truecaser.perl --corpus data/$train.tok.clean.$TRG --model model/truecase-model.$TRG &
wait
# apply truecaser (cleaned training train)
for prefix in $train
 do
  $mosesdecoder/scripts/recaser/truecase.perl --model model/truecase-model.$SRC < data/$prefix.tok.clean.$SRC > data/$prefix.tc.$SRC &
  $mosesdecoder/scripts/recaser/truecase.perl --model model/truecase-model.$TRG < data/$prefix.tok.clean.$TRG > data/$prefix.tc.$TRG &
 done
wait
# apply truecaser (dev/test files)
for prefix in $devel
 do
  $mosesdecoder/scripts/recaser/truecase.perl --model model/truecase-model.$SRC < data/$prefix.tok.$SRC > data/$prefix.tc.$SRC &
  $mosesdecoder/scripts/recaser/truecase.perl --model model/truecase-model.$TRG < data/$prefix.tok.$TRG > data/$prefix.tc.$TRG &
 done
wait

# train BPE
cat data/$train.tc.$SRC data/$train.tc.$TRG | $subword_nmt/learn_bpe.py -s $bpe_operations > model/$SRC$TRG.bpe

# apply BPE

for prefix in $train $devel
 do
  python $subword_nmt/apply_bpe.py -c model/$SRC$TRG.bpe < data/$prefix.tc.$SRC > data/$prefix.bpe.$bpe_operations.$SRC &
  python $subword_nmt/apply_bpe.py -c model/$SRC$TRG.bpe < data/$prefix.tc.$TRG > data/$prefix.bpe.$bpe_operations.$TRG &
 done
wait

# build network dictionary
PYTHONPATH=../ python -m bin.build_vocab data/train.bpe.$bpe_operations.$SRC data/train.bpe.$bpe_operations.$TRG --save_vocab data/vocab.bpe.$bpe_operations.$SRC$TRG