@echo off
echo ============================================================
echo Installing Python Dependencies for Dance Competition Analysis
echo ============================================================
echo.

echo [1/11] Installing pandas...
pip install pandas -q

echo [2/11] Installing numpy...
pip install numpy -q

echo [3/11] Installing tqdm...
pip install tqdm -q

echo [4/11] Installing PuLP (Integer Programming)...
pip install PuLP -q

echo [5/11] Installing PyMC3 (Bayesian Modeling)...
pip install pymc3 -q

echo [6/11] Installing arviz...
pip install arviz -q

echo [7/11] Installing scipy...
pip install scipy -q

echo [8/11] Installing hmmlearn (Hidden Markov Models)...
pip install hmmlearn -q

echo [9/11] Installing scikit-learn...
pip install scikit-learn -q

echo [10/11] Installing matplotlib...
pip install matplotlib -q

echo [11/11] Installing seaborn...
pip install seaborn -q

echo.
echo ============================================================
echo Installation completed!
echo ============================================================
echo.
echo You can now run:
echo   python test_data_load.py           (Test data loading)
echo   python dance_competition_analysis_quick_test.py  (Quick test)
echo   python dance_competition_analysis.py   (Full analysis)
echo.
pause
