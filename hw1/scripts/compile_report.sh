#!/bin/bash
# 编译 LaTeX 报告

echo "编译 LaTeX 报告..."

cd report

pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex

echo "报告编译完成: report/report.pdf"
