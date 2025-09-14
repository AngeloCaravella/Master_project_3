@echo off
REM Naviga nella cartella del repository
cd /d "C:\Users\angel\OneDrive\Desktop\Project_Master"

REM Aggiunge tutti i cambiamenti
git add .

REM Commit con messaggio automatico contenente data e ora
set MSG="Aggiornamento automatico %date% %time%"
git commit -m %MSG%

REM Push sul branch principale (main)
git push origin main

pause
