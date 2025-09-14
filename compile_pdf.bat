@echo off
REM ====================================================
REM Batch file to compile LaTeX with Biber automatically
REM ====================================================

REM Step 1: Compile LaTeX to generate .bcf
pdflatex -interaction=nonstopmode main.tex

REM Step 2: Run Biber
biber main

REM Step 3: Compile LaTeX again to include bibliography
pdflatex -interaction=nonstopmode main.tex

REM Step 4: Compile LaTeX one more time for cross-references
pdflatex -interaction=nonstopmode main.tex

REM Done
echo Compilation finished!
pause
