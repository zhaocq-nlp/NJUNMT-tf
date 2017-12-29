
scriptDir="./mteval"
srcSgmFile="./mteval/testdata/newstest2017-enzh-src.en.sgm"
trgTokSgmFile="./mteval/testdata/newstest2017-enzh-ref.zh.tok.sgm"
transOriginFile="./mteval/testdata/original"

python -u ${scriptDir}/mteval.py -s ${srcSgmFile} -t ${transOriginFile} -r ${trgTokSgmFile} -d ${scriptDir}
