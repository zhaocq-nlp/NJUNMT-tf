# Copyright 2017 ZhaoChengqi, zhaocq@nlp.nju.edu.cn, Natural Language Processing Group, Nanjing University.
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
