#!/usr/bin/env python

import re
import sys

re_space = re.compile(r"(?<![a-zA-Z])\s(?![a-zA-Z])", flags=re.UNICODE)
re_final_comma = re.compile("\.$")

for line in sys.stdin:
  line = line[:-1]
  line = re_space.sub("", line.decode("utf8"))
  line = line.replace(",", u"\uFF0C")
  line = re_final_comma.sub(u"\u3002", line)
  print line.encode("utf8")
