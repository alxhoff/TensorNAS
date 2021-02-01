#!/bin/bash


python $1.py
pdflatex $2.tex

rm *.aux *.log *.vscodeLog
#rm *.tex

if [[ "$OSTYPE" == "Darwin"* ]]; then
    open $2.pdf
else
    open $2.pdf
fi