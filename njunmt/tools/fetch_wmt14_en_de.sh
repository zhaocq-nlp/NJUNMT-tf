#!/usr/bin/env bash
# Copyright 2017 Natural Language Processing Group, Nanjing University, zhaocq.nlp@gmail.com.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# refer to https://github.com/tensorflow/nmt/blob/master/nmt/scripts/wmt16_en_de.sh

set -e

MOSES_DIR=~/Documents/gitdownload/mosesdecoder
BPE_DIR=~/Documents/gitdownload/subword-nmt

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

OUTPUT_DIR="${1:-wmt14_de_en}"
echo "Writing to ${OUTPUT_DIR}. To change this, set the OUTPUT_DIR environment variable."

OUTPUT_DIR_DATA="${OUTPUT_DIR}/data"
mkdir -p $OUTPUT_DIR_DATA

echo "Downloading Europarl v7. This may take a while..."
curl -o ${OUTPUT_DIR_DATA}/europarl-v7-de-en.tgz \
  http://www.statmt.org/europarl/v7/de-en.tgz

echo "Downloading Common Crawl corpus. This may take a while..."
curl -o ${OUTPUT_DIR_DATA}/common-crawl.tgz \
  http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz

echo "Downloading News Commentary v9. This may take a while..."
curl -o ${OUTPUT_DIR_DATA}/nc-v9.tgz \
  http://www.statmt.org/wmt14/training-parallel-nc-v9.tgz

echo "Downloading dev/test sets"
curl -o ${OUTPUT_DIR_DATA}/dev.tgz \
  http://www.statmt.org/wmt14/dev.tgz
curl -o ${OUTPUT_DIR_DATA}/test.tgz \
  http://www.statmt.org/wmt14/test-full.tgz


# Extract everything
echo "Extracting all files..."
mkdir -p "${OUTPUT_DIR_DATA}/europarl-v7-de-en"
tar -xvzf "${OUTPUT_DIR_DATA}/europarl-v7-de-en.tgz" -C "${OUTPUT_DIR_DATA}/europarl-v7-de-en"
mkdir -p "${OUTPUT_DIR_DATA}/common-crawl"
tar -xvzf "${OUTPUT_DIR_DATA}/common-crawl.tgz" -C "${OUTPUT_DIR_DATA}/common-crawl"
mkdir -p "${OUTPUT_DIR_DATA}/nc-v9"
tar -xvzf "${OUTPUT_DIR_DATA}/nc-v9.tgz" -C "${OUTPUT_DIR_DATA}/nc-v9"
mkdir -p "${OUTPUT_DIR_DATA}/dev"
tar -xvzf "${OUTPUT_DIR_DATA}/dev.tgz" -C "${OUTPUT_DIR_DATA}/dev"
mkdir -p "${OUTPUT_DIR_DATA}/test"
tar -xvzf "${OUTPUT_DIR_DATA}/test.tgz" -C "${OUTPUT_DIR_DATA}/test"


# Concatenate Training data
cat "${OUTPUT_DIR_DATA}/europarl-v7-de-en/training/europarl-v7.de-en.en" \
  "${OUTPUT_DIR_DATA}/common-crawl/commoncrawl.de-en.en" \
  "${OUTPUT_DIR_DATA}/nc-v9/training/news-commentary-v9.de-en.en" \
  > "${OUTPUT_DIR}/train.en"
wc -l "${OUTPUT_DIR}/train.en"


cat "${OUTPUT_DIR_DATA}/europarl-v7-de-en/training/europarl-v7.de-en.de" \
  "${OUTPUT_DIR_DATA}/common-crawl/commoncrawl.de-en.de" \
  "${OUTPUT_DIR_DATA}/nc-v9/training/news-commentary-v9.de-en.de" \
  > "${OUTPUT_DIR}/train.de"
wc -l "${OUTPUT_DIR}/train.de"


# Clone Moses
if [ ! -d "${MOSES_DIR}" ]; then
  echo "Cloning moses for data processing"
  git clone https://github.com/moses-smt/mosesdecoder.git "${MOSES_DIR}"
fi

# Convert newstest2014 data into raw text format
${MOSES_DIR}/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/test/test-full/newstest2014-deen-src.de.sgm \
  > ${OUTPUT_DIR_DATA}/test/test-full/newstest2014.deen.de
${MOSES_DIR}/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/test/test-full/newstest2014-deen-ref.en.sgm \
  > ${OUTPUT_DIR_DATA}/test/test-full/newstest2014.deen.en
${MOSES_DIR}/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/test/test-full/newstest2014-deen-src.en.sgm \
  > ${OUTPUT_DIR_DATA}/test/test-full/newstest2014.ende.en
${MOSES_DIR}/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/test/test-full/newstest2014-deen-ref.de.sgm \
  > ${OUTPUT_DIR_DATA}/test/test-full/newstest2014.ende.de

# Convert newstest2013 data into raw text format
${MOSES_DIR}/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/dev/dev/newstest2013-src.de.sgm \
  > ${OUTPUT_DIR_DATA}/dev/dev/newstest2013.deen.de
${MOSES_DIR}/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/dev/dev/newstest2013-ref.en.sgm \
  > ${OUTPUT_DIR_DATA}/dev/dev/newstest2013.deen.en
${MOSES_DIR}/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/dev/dev/newstest2013-src.en.sgm \
  > ${OUTPUT_DIR_DATA}/dev/dev/newstest2013.ende.en
${MOSES_DIR}/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/dev/dev/newstest2013-ref.de.sgm \
  > ${OUTPUT_DIR_DATA}/dev/dev/newstest2013.ende.de

# Convert newstest2012 data into raw text format
${MOSES_DIR}/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/dev/dev/newstest2012-src.de.sgm \
  > ${OUTPUT_DIR_DATA}/dev/dev/newstest2012.deen.de
${MOSES_DIR}/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/dev/dev/newstest2012-ref.en.sgm \
  > ${OUTPUT_DIR_DATA}/dev/dev/newstest2012.deen.en
${MOSES_DIR}/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/dev/dev/newstest2012-src.en.sgm \
  > ${OUTPUT_DIR_DATA}/dev/dev/newstest2012.ende.en
${MOSES_DIR}/scripts/ems/support/input-from-sgm.perl \
  < ${OUTPUT_DIR_DATA}/dev/dev/newstest2012-ref.de.sgm \
  > ${OUTPUT_DIR_DATA}/dev/dev/newstest2012.ende.de

# Copy dev/test data to output dir
cp ${OUTPUT_DIR_DATA}/test/test-full/newstest2014.deen* ${OUTPUT_DIR}/
cp ${OUTPUT_DIR_DATA}/test/test-full/newstest2014.ende* ${OUTPUT_DIR}/
cp ${OUTPUT_DIR_DATA}/dev/dev/newstest2013.deen* ${OUTPUT_DIR}/
cp ${OUTPUT_DIR_DATA}/dev/dev/newstest2013.ende* ${OUTPUT_DIR}/
cp ${OUTPUT_DIR_DATA}/dev/dev/newstest2012.deen* ${OUTPUT_DIR}/
cp ${OUTPUT_DIR_DATA}/dev/dev/newstest2012.ende* ${OUTPUT_DIR}/

cat ${OUTPUT_DIR}/newstest2012.ende.en ${OUTPUT_DIR}/newstest2013.ende.en > ${OUTPUT_DIR}/dev.ende.en
cat ${OUTPUT_DIR}/newstest2012.ende.de ${OUTPUT_DIR}/newstest2013.ende.de > ${OUTPUT_DIR}/dev.ende.de
cat ${OUTPUT_DIR}/newstest2012.deen.de ${OUTPUT_DIR}/newstest2013.deen.de > ${OUTPUT_DIR}/dev.deen.de
cat ${OUTPUT_DIR}/newstest2012.deen.en ${OUTPUT_DIR}/newstest2013.deen.en > ${OUTPUT_DIR}/dev.deen.en

# Tokenize data
for f in ${OUTPUT_DIR}/*.de; do
  echo "Tokenizing $f..."
  ${MOSES_DIR}/scripts/tokenizer/tokenizer.perl -q -l de -threads 8 -no-escape < $f > ${f%.*}.tok.de
done


for f in ${OUTPUT_DIR}/*.en; do
  echo "Tokenizing $f..."
  ${MOSES_DIR}/scripts/tokenizer/tokenizer.perl -q -l en -threads 8 -no-escape < $f > ${f%.*}.tok.en
done

# Clean train corpora by length
for f in ${OUTPUT_DIR}/train.tok.en; do
  fbase=${f%.*}
  echo "Cleaning ${fbase}..."
  ${MOSES_DIR}/scripts/training/clean-corpus-n.perl $fbase de en "${fbase}.clean" 1 100
done


# Generate Subword Units (BPE)
# Clone Subword NMT
if [ ! -d "${BPE_DIR}" ]; then
  git clone https://github.com/rsennrich/subword-nmt.git "${BPE_DIR}"
fi


# Learn Shared BPE
for merge_ops in 32000; do
  echo "Learning BPE with merge_ops=${merge_ops}. This may take a while..."
  cat "${OUTPUT_DIR}/train.tok.clean.de" "${OUTPUT_DIR}/train.tok.clean.en" | \
    ${BPE_DIR}/learn_bpe.py -s $merge_ops > "${OUTPUT_DIR}/bpe.${merge_ops}"

  echo "Apply BPE with merge_ops=${merge_ops} to tokenized files..."
  for lang in en de; do
    for f in ${OUTPUT_DIR}/*.tok.${lang} ${OUTPUT_DIR}/*.tok.clean.${lang}; do
      outfile="${f%.*}.bpe.${merge_ops}.${lang}"
      ${BPE_DIR}/apply_bpe.py -c "${OUTPUT_DIR}/bpe.${merge_ops}" < $f > "${outfile}"
      echo ${outfile}
    done
  done

done

rm -r ${OUTPUT_DIR_DATA}

echo "All done."
