#/bin/sh

# suffix of source language files
NOW=$1
output_file=$2


# number of merge operations. Network vocabulary should be slightly larger (to include characters),
# or smaller if the operations are learned on the joint vocabulary
bpe_operations=40000

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=../third_party/mosesdecoder

# path to subword segmentation scripts: https://github.com/rsennrich/subword-nmt
subword_nmt=../third_party

$mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -l $NOW | \
$mosesdecoder/scripts/tokenizer/tokenizer.perl -a -l $NOW -threads 10 |
$mosesdecoder/scripts/recaser/truecase.perl --model model/truecase-model.$NOW > /tmp/tmp.txt

PYTHONPATH=../ python -m bin.prepend_language_to_corpus --corpora data/test_big.tc.$NOW --prefixes $NOW

if [ -z "${output_file}" ];  then
  mv /tmp/tmp.txt output_file
else
  cat /tmp/tmp.txt
  rm /tmp/tmp.txt
fi

#python $subword_nmt/apply_bpe.py -c model/$SRC$TRG.bpe --vocabulary model/vocab.bpe.$bpe_operations.$NOW --vocabulary-threshold 50
