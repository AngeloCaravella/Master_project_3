@echo off
REM Prende il file main.tex nella cartella corrente
set TEXFILE="main.tex"

REM Cartella di output per il PDF (puoi cambiarla se vuoi)
set OUTPUTDIR="PDF_Output"

REM Crea la cartella di output se non esiste
if not exist %OUTPUTDIR% (
    mkdir %OUTPUTDIR%
)

REM Compila il file .tex usando pdflatex e salva il PDF nella cartella di output
pdflatex -output-directory=%OUTPUTDIR% %TEXFILE%

REM Mostra il messaggio di completamento
echo Compilazione completata! PDF salvato in %OUTPUTDIR%
pause
