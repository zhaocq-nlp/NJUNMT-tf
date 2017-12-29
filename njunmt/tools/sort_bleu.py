import numpy
import sys

with open(sys.argv[1], "r") as fp, \
    open(sys.argv[1] + ".sort", "w") as fw:

    line_list = []
    bleu_list = []
    for line in fp:
        if "BLEU=" not in line:
            continue
        line = line.strip()
        line_list.append(line)
        idx = line.index("BLEU=")
        bleu_list.append(float(line[idx+5:idx+10]))

    argidx = numpy.argsort(bleu_list)
    fw.write("\n".join([line_list[idx] for idx in argidx[::-1]]))
