#!/usr/bin/env bash

SCRIPT_DIR=njunmt/tools/mteval
ORGANIZATION="simpletest"

HELP="0"
CHAR_LEVEL="0"
DETOKENIZE="0"
VERBOSE="0"

while getopts "s:r:t:l:o:hcda" opt
do
    case $opt in
        s )
            SRC_SGMFILE=$OPTARG;;
        r )
            REF_SGMFILE=$OPTARG;;
        t )
            TRANS_PLAINFILE=$OPTARG;;
        l )
            LANGUAGE=$OPTARG;;
        o )
            ORGANIZATION=$OPTARG;;
        h )
            HELP="1";;
        c )
            CHAR_LEVEL="1";;
        d )
            DETOKENIZE="1";;
        a )
            VERBOSE="1";;
    esac
done

if [ "$HELP" -eq "1" ]; then
    echo "Usage: sh mteval_wrapper.sh -s SOURCE_SGM -r REF_SGM -t TRANS_PLAIN -l [en|de|zh|...] -o ORGANIZATION"
    echo "Options:"
    echo "  -s      ... Source SGM file (XML format)."
    echo "  -r      ... Reference SGM file (XML format)."
    echo "  -t      ... Translation file (plain text)."
    echo "  -l      ... Language."
    echo "  -o      ... Organization."
    echo "  -c      ... Flag, whether to convert to character level (for Chinese only)."
    echo "  -d      ... Flag, whether to detokenize the translation file."
    echo "  -a      ... Flag, whether to display all information from mteval."
    echo "  -h      ... Show help information."
    exit 0
fi

if [ ! -e ${SRC_SGMFILE} ]; then
    echo "source file ${SRC_SGMFILE} not exists."
    exit 1
fi

if [ ! -e ${REF_SGMFILE} ]; then
    echo "reference file ${REF_SGMFILE} not exists."
    exit 1
fi

if [ ! -e ${TRANS_PLAINFILE} ]; then
    echo "translation file ${TRANS_PLAINFILE} not exists."
    exit 1
fi


# to character level
if [ "$CHAR_LEVEL" -eq "1" ]; then
    if [ "$DETOKENIZE" -eq "1" ]; then
        echo "-c and -d options can not be enabled together."
        exit 1
    fi
    NEW_REF_SGMFILE=${REF_SGMFILE%.*}.$((`date +%s`)).${REF_SGMFILE##*.}
    python ${SCRIPT_DIR}/tokenizeChinese.py ${REF_SGMFILE} ${NEW_REF_SGMFILE}
    REF_SGMFILE=${NEW_REF_SGMFILE}

    NEW_TRANS_PLAINFILE=${TRANS_PLAINFILE}.$((`date +%s`))
    python ${SCRIPT_DIR}/tokenizeChinese.py ${TRANS_PLAINFILE} ${NEW_TRANS_PLAINFILE}
    TRANS_PLAINFILE=${NEW_TRANS_PLAINFILE}
fi

if [ "$LANGUAGE" != "zh" ]; then
    if [ "$LANGUAGE" != "en" ]; then
        if [ "$LANGUAGE" != "de" ]; then
            echo "Error with language $LANGUAGE."
            echo "Now this script only accepts en, de, zh."
            exit 1
        fi
    fi
    NEW_TRANS_PLAINFILE=${TRANS_PLAINFILE}.$((`date +%s`))
    perl ${SCRIPT_DIR}/detruecase.perl < ${TRANS_PLAINFILE} > ${NEW_TRANS_PLAINFILE}
    if [ "$DETOKENIZE" -eq "1" ]; then
        mv ${NEW_TRANS_PLAINFILE} ${TRANS_PLAINFILE}
    else
        TRANS_PLAINFILE=${NEW_TRANS_PLAINFILE}
    fi
fi

# detokenize
if [ "$DETOKENIZE" -eq "1" ]; then
    NEW_TRANS_PLAINFILE=${TRANS_PLAINFILE}.detok.$((`date +%s`))
    perl ${SCRIPT_DIR}/detokenizer.perl -l ${LANGUAGE} -q < ${TRANS_PLAINFILE} > ${NEW_TRANS_PLAINFILE}
    TRANS_PLAINFILE=${NEW_TRANS_PLAINFILE}
fi

# wrap xml
TRANS_SGMFILE=${TRANS_PLAINFILE}.sgm
perl ${SCRIPT_DIR}/wrap-xml.perl ${LANGUAGE} ${SRC_SGMFILE} ${ORGANIZATION} < ${TRANS_PLAINFILE} > ${TRANS_SGMFILE}

# run mteval
if [ "$VERBOSE" -eq "1" ]; then
    perl ${SCRIPT_DIR}/mteval-v13a.pl -s ${SRC_SGMFILE} -r ${REF_SGMFILE} -t ${TRANS_SGMFILE}
else
    RES=`perl ${SCRIPT_DIR}/mteval-v13a.pl -s ${SRC_SGMFILE} -r ${REF_SGMFILE} -t ${TRANS_SGMFILE}`
    BLEU=${RES#*BLEU score = }
    echo "${BLEU%% *}"
fi

if [ "$CHAR_LEVEL" -eq "1" ]; then
    rm ${TRANS_PLAINFILE}
    rm ${REF_SGMFILE}
fi
if [ "$DETOKENIZE" -eq "1" ]; then
    rm ${TRANS_PLAINFILE}
fi
