# -*- coding:utf-8 -*-
import os
import sys
import argparse
import subprocess
import time

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # source sgm
    parser.add_argument('-s', '--source', type=str)
    # reference sgm
    parser.add_argument('-r', '--reference', type=str)
    # translations
    parser.add_argument('-t', '--translation', type=str)
    # script dir
    parser.add_argument('-d', '--script-dir', type=str)

    args = parser.parse_args()

    # to character level
    trans_tok = args.translation + '.parsed.tok' + str(time.time())
    os.system('python %s %s %s'
              % (os.path.join(args.script_dir, 'tokenizeChinese.py'), args.translation, trans_tok))

    # transfer to sgm type
    trans_tok_sgm = trans_tok +  '.sgm'
    os.system('perl %s zh %s toutiaoAI < %s > %s'
              % (os.path.join(args.script_dir, 'wrap-xml.perl'), args.source,
                 trans_tok, trans_tok_sgm))

    # bleu
    cmd = 'perl %s -s %s -r %s -t %s' \
          % (os.path.join(args.script_dir, 'mteval-v13a.pl'),
             args.source, args.reference, trans_tok_sgm)

    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    popen.wait()
    ret = ''
    for line in popen.stdout.readlines():
        if line.find('BLEU') >= 0:
            ret = line.strip()
            break
    print ret

    os.system('rm %s %s' % (trans_tok_sgm, trans_tok))



