from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import os
from pathlib import Path

app = FastAPI(title="Forecasting System API")

# 1. CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Pydantic Models
class ForecastEntry(BaseModel):
    date: str
    predicted_sales: float

class ForecastResponse(BaseModel):
    state: str
    weeks: int
    forecast: List[ForecastEntry]

class ModelInfo(BaseModel):
    best_model: str
    rmse: float
    mae: Optional[float] = None
    mape: Optional[float] = None

class StatesResponse(BaseModel):
    states: List[str]

class MonitorRequest(BaseModel):
    state: str
    actuals: List[float]
    threshold_mape: float = 25.0

# 3. File Paths
FORECASTS_PATH = Path("data/forecasts.json")
REGISTRY_PATH = Path("model_registry.json")

# 4. Startup Check
@app.on_event("startup")
async def startup_event():
    print(f"Forecasting API is running.")
    print(f"Docs available at: http://127.0.0.1:8000/docs (Note: Port may vary if started via run_pipeline.py)")
    
    if not FORECASTS_PATH.exists():
        print("WARNING: data/forecasts.json is missing. API will return 404s for forecasts.")
    if not REGISTRY_PATH.exists():
        print("WARNING: model_registry.json is missing. API will return empty model lists.")

# 5. Endpoints
@app.get("/")
async def root():
    return {"message": "Forecasting API is running. Visit /dashboard for UI."}

@app.get("/dashboard")
async def get_dashboard():
    """
    Serves the interactive forecasting dashboard.
    """
    path = Path(__file__).parent / "dashboard.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Dashboard file not found")
    return FileResponse(path)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """
    Returns an empty response for favicon to prevent 404 errors in the dashboard.
    """
    return Response(status_code=204)

@app.get("/states", response_model=StatesResponse)
async def get_states():
    if not REGISTRY_PATH.exists():
        return {"states": []}
    
    with open(REGISTRY_PATH, "r") as f:
        registry = json.load(f)
    return {"states": [s for s in registry.keys() if not s.startswith('_')]}

@app.get("/models", response_model=Dict[str, ModelInfo])
async def get_models():
    if not REGISTRY_PATH.exists():
        return {}
    
    with open(REGISTRY_PATH, "r") as f:
        registry = json.load(f)
    
    # Extract only the summary info for each state
    output = {}
    for state, info in registry.items():
        if state.startswith('_'):
            continue
        output[state] = ModelInfo(
            best_model=info['best_model'],
            rmse=info['rmse'],
            mae=info.get('mae'),
            mape=info.get('mape')
        )
    return output

@app.get("/forecast", response_model=ForecastResponse)
async def get_forecast(state: str, weeks: int = Query(8, ge=1, le=52)):
    if not FORECASTS_PATH.exists():
        raise HTTPException(status_code=500, detail="Forecast data source not found.")
        
    with open(FORECASTS_PATH, "r") as f:
        all_forecasts = json.load(f)
        
    if state not in all_forecasts:
        raise HTTPException(status_code=404, detail="State not found")
        
    state_forecast = all_forecasts[state]
    # Slice to requested weeks
    requested_forecast = state_forecast[:weeks]
    
    return {
        "state": state,
        "weeks": len(requested_forecast),
        "forecast": requested_forecast
    }

@app.post("/monitor")
async def monitor_state(request: MonitorRequest):
    """
    Compares actual observed sales against stored forecasts.
    Triggers retraining if MAPE exceeds threshold.
    """
    # Dynamic import to avoid issues if monitor.py has circular dependencies
    try:
        from src.monitor import check_accuracy_drift
    except ImportError:
        import sys
        sys.path.append(os.path.join(os.getcwd(), 'src'))
        from monitor import check_accuracy_drift
        
    try:
        result = check_accuracy_drift(
            request.state,
            request.actuals,
            request.threshold_mape
        )
        if result is None:
             raise HTTPException(status_code=404, detail=f"State {request.state} not found or no forecast exists.")
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
