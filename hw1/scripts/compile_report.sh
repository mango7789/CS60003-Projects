#!/bin/bash
# 编译 LaTeX 报告
# 使用方法: ./scripts/compile_report.sh

REPORT_DIR="$(cd "$(dirname "$0")/../report" && pwd)"

echo "编译 LaTeX 报告..."
cd "$REPORT_DIR"

# 编译两次以生成目录
pdflatex -interaction=nonstopmode report.tex
pdflatex -interaction=nonstopmode report.tex

echo "报告编译完成: $REPORT_DIR/report.pdf"
