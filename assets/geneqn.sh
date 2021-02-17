#!/usr/bin/bash
cat ./modelv1.tex | pnglatex -o ./modelv1.png -O -s 12 -d 300 -e align\* -p mathtools,amssymb,amsthm
