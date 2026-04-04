@echo off
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting QuantumPy web app...
streamlit run app.py
pause
