@echo off

IF EXIST venv_312 GOTO has_venv_312
call py -3.12 -m virtualenv -p "C:\Program Files\Python312\python.exe" venv_312
:has_venv_312
call venv_312\Scripts\pip.exe install -r requirements.txt
call venv_312\Scripts\activate.bat

chcp 65001

python thermal_picture_analysis.py
PAUSE