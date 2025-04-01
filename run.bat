@echo off

:: Activate the conda environment for Prism-AI-Studio
CALL "C:\ProgramData\<your anaconda distribution name>\Scripts\activate.bat" Prism-AI-Studio

:: Navigate to the Prism-AI-Studio directory (Change path according to yourself)
cd /D path/to/your/Prism-AI-Studio

:: Run the Prism-AI-Studio web interface script
python prism.py