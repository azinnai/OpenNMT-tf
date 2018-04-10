#/bin/sh

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=~/nmt/OpenNMT-tf/third_party/mosesdecoder

# suffix of target language files
lng=it

sed 's/\@\@ //g' | \
$mosesdecoder/scripts/recaser/detruecase.perl | \
sed -e 's/ <\/s>//g'
