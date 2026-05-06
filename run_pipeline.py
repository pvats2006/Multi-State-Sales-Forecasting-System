import subprocess
import sys
import time
import os
import socket
import json
import importlib.util
from pathlib import Path

def check_dependencies():
    """
    Performs a pre-flight check of Python version and required packages.
    """
    print(f"\n{'='*50}")
    print("PRE-FLIGHT CHECK")
    print(f"{'='*50}")
    
    # 1. Check Python Version
    py_version = sys.version_info
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 8):
        print(f"[FAIL] Python 3.8+ required. Found {py_version.major}.{py_version.minor}")
        sys.exit(1)
    print(f"[OK] Python {py_version.major}.{py_version.minor} detected.")

    # 2. Check Packages
    packages = {
        "pandas": "pandas",
        "numpy": "numpy",
        "statsmodels": "statsmodels",
        "pmdarima": "pmdarima",
        "prophet": "prophet",
        "xgboost": "xgboost",
        "sklearn": "scikit-learn",
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "joblib": "joblib",
        "holidays": "holidays"
    }
    
    all_ok = True
    for module, p_name in packages.items():
        spec = importlib.util.find_spec(module)
        if spec is None:
            print(f"[FAIL] {p_name} is missing. Run: pip install {p_name}")
            all_ok = False
        else:
            print(f"[OK] {p_name} is installed.")

    # Check TensorFlow (Optional)
    tf_spec = importlib.util.find_spec("tensorflow")
    if tf_spec is None:
        print("[!] tensorflow is missing (LSTM model will be skipped).")
    else:
        print("[OK] tensorflow is installed.")

    if not all_ok:
        print("\n[ERROR] Missing dependencies. Please install them before continuing.")
        sys.exit(1)
    
    print(f"{'='*50}\nAll essential dependencies verified.\n")

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def find_free_port(start=8000, end=8010):
    for port in range(start, end):
        if not is_port_in_use(port):
            return port
    raise RuntimeError(f"No free ports found between {start} and {end}")

def run_step(command, description):
    print(f"\n{'='*50}")
    print(f"STEP: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*50}")
    
    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{description}' failed. Exiting.")
        sys.exit(1)
    
    print(f"[SUCCESS] {description} completed.")

def print_model_summary():
    """
    Reads model_registry.json and prints an enhanced summary table.
    """
    registry_path = Path("model_registry.json")
    if not registry_path.exists():
        return
        
    with open(registry_path, "r") as f:
        registry = json.load(f)
        
    print("\n" + "="*80)
    print(f"{'State':<15} | {'Best Model':<10} | {'RMSE':<10} | {'MAPE %':<8} | {'Models Tried'}")
    print("-" * 80)
    for state, info in registry.items():
        if state.startswith('_'):
            continue
        models_tried = ", ".join(info.get('all_models', {}).keys())
        print(f"{state:<15} | {info['best_model']:<10} | {info['rmse']:<10.2f} | {info['mape']:<8.2f} | {models_tried}")
    print("="*80)

def print_all_forecasts():
    """
    Reads data/forecasts.json and prints the full 8-week forecast for every state.
    """
    forecast_path = Path("data/forecasts.json")
    if not forecast_path.exists():
        return
        
    with open(forecast_path, "r") as f:
        forecasts = json.load(f)
        
    print("\nPRODUCTION FORECASTS (Next 8 Weeks):")
    for state, entries in forecasts.items():
        print(f"\n[{state}]")
        print(f"  {'Date':<12} | {'Predicted Sales':<15}")
        print(f"  {'-'*29}")
        for entry in entries:
            print(f"  {entry['date']:<12} | {entry['predicted_sales']:<15.2f}")

def main():
    # 0. Pre-flight Check
    check_dependencies()
    
    print("Starting Forecasting Pipeline...")
    
    # 1. Preprocess
    run_step("python src/preprocess.py", "Data Preprocessing")
    
    # 2. Evaluate
    run_step("python src/evaluate.py", "Model Training & Evaluation")
    print_model_summary()
    
    # 3. Forecast
    run_step("python src/forecast.py", "Generating Production Forecasts")
    print_all_forecasts()
    
    # 4. Start API Server
    port = find_free_port(8000, 8010)
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETE")
    print("="*50)
    print(f"API URL  : http://127.0.0.1:{port}")
    print(f"Docs     : http://127.0.0.1:{port}/docs")
    print("Endpoints:")
    print(f"  GET /forecast?state=California&weeks=8")
    print(f"  GET /models")
    print(f"  GET /states")
    print("="*50)
    
    try:
        subprocess.run(f"uvicorn api.main:app --host 0.0.0.0 --port {port}", shell=True)
    except KeyboardInterrupt:
        print("\nPipeline stopped by user.")

if __name__ == "__main__":
    main()
