#!/usr/bin/bash
cat ./modelv1.tex | pnglatex -o ./modelv1.png -O -s 12 -d 300 -e align\* -p mathtools,amssymb,amsthm
cat ./modelv1_simplified.tex | pnglatex -o ./modelv1_simplified.png -O -s 12 -d 300 -e align\* -p mathtools,amssymb,amsthm
