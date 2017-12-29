from __future__ import print_function
import sys
import numpy


def shuffle_data(from_binding, to_binding):
    lines_list = []
    fps = []
    fws = []
    for idx in range(len(from_binding)):
        lines_list.append([])
        fps.append(open(from_binding[idx], "r"))

    for zip_lines in zip(*fps):
        for idx in range(len(zip_lines)):
            lines_list[idx].append(zip_lines[idx].strip())
    for fp in fps:
        fp.close()
    for idx in range(len(to_binding)):
        fws.append(open(to_binding[idx], "w"))
    rands = numpy.arange(len(lines_list[0]))
    numpy.random.shuffle(rands)
    for i in rands:
        for idx in range(len(lines_list)):
            fws[idx].write(lines_list[idx][i] + "\n")
    for fw in fws:
        fw.close()


froms = sys.argv[1]
tos = sys.argv[2]

shuffle_data(froms.strip().split(","), tos.strip().split(","))
