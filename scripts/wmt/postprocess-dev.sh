#/bin/sh

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=~/nmt/OpenNMT-tf/third_party/mosesdecoder

sed 's/\@\@ //g' | \
$mosesdecoder/scripts/recaser/detruecase.perl | \
sed -e 's/ <\/s>//g' | sed -e 's/[a-z,A-Z]*_//g'
