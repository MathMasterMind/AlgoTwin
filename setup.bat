
:: Set variables
set "URL=https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_11.3.1_build/ghidra_11.3.1_PUBLIC_20250219.zip"
set "ZIPFILE=ghidra_11.3.1_PUBLIC_20250219.zip"

:: Download using PowerShell
echo Downloading %ZIPFILE%...
if not exist "%ZIPFILE%" powershell -Command "Invoke-WebRequest -Uri '%URL%' -OutFile '%ZIPFILE%'"

:: Extract using PowerShell
echo Extracting %ZIPFILE%...
powershell -Command "Expand-Archive -LiteralPath '%ZIPFILE%' -Force"

:: Make LocalConfig.py
echo Creating LocalConfig.py
echo GHIDRA_INSTALL_PATH="%CD%\ghidra_11.3.1_PUBLIC_20250219" > LocalConfig.py

:: Set up Python virtual environment
python3 -m venv venv
call "venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -i https://download.pytorch.org/whl/cu126 torch torchvision torchaudio 
pip install python-magic pyghidra pandas python-Levenshtein torch_geometric python-magic-bin setuptools wheel
pip install --no-use-pep517 torch_scatter

@REM :: Download the dataset
@REM set "URL=https://github.com/MathMasterMind/ECE6254-Project/releases/download/Training/Binaries.zip"
@REM set "ZIPFILE=Binaries.zip"

@REM echo Downloading %ZIPFILE%...
@REM if not exist "%ZIPFILE%" powershell -Command "Invoke-WebRequest -Uri '%URL%' -OutFile '%ZIPFILE%'"

@REM echo Extracting %ZIPFILE%...
@REM powershell -Command "Expand-Archive -LiteralPath '%ZIPFILE%' -Force"