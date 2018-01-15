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
import sys
import numpy

print "shuffling data...."
with open(sys.argv[1], 'r') as fzh, open(sys.argv[2], 'r') as fen, \
    open(sys.argv[3], 'w') as fwzh, open(sys.argv[4], 'w') as fwen:
    lines_zh = []
    lines_en = []
    for zh, en in zip(fzh, fen):
        lines_zh.append(zh)
        lines_en.append(en)

    rands = numpy.arange(len(lines_zh)) # numpy.random.randint(len(lines_zh), size=len(lines_zh))
    numpy.random.shuffle(rands)
    for i in rands:
        fwzh.write(lines_zh[i])
        fwen.write(lines_en[i])
