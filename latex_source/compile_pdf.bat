@echo off
REM ====================================================
REM Batch file to compile LaTeX with Biber automatically
REM This script assumes it is run from within the same
REM directory as the main.tex file.
REM ====================================================

REM Step 1: Compile LaTeX to generate .bcf
pdflatex -interaction=nonstopmode main.tex

REM Step 2: Run Biber
biber main

REM Step 3: Compile LaTeX again to include bibliography
pdflatex -interaction=nonstopmode main.tex

REM Step 4: Compile LaTeX one more time for cross-references
pdflatex -interaction=nonstopmode main.tex

REM Step 5: Move the final PDF to the parent directory
move /Y main.pdf ..\main.pdf

REM Done
echo Compilation finished in latex_source directory!
pause
