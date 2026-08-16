import os
import re
import json
from typing import Optional, Dict, Any
from pydantic import BaseModel
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

app = FastAPI(title="FloatChat AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

try:
    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except Exception:
    groq_client = None

class ChatRequest(BaseModel):
    message: str

def generate_ocean_profile(region: str = "Arabian Sea", max_depth: float = 1000.0) -> pd.DataFrame:
    depths = np.linspace(5, max_depth, 60)
    if "bengal" in region.lower():
        surface_temp, surface_psal = 29.5, 33.2
    elif "atlantic" in region.lower():
        surface_temp, surface_psal = 22.0, 36.5
    elif "pacific" in region.lower():
        surface_temp, surface_psal = 26.0, 34.8
    else:
        surface_temp, surface_psal = 28.5, 36.2

    temp = (surface_temp - 2.0) * np.exp(-depths / 240.0) + 2.0 + np.random.normal(0, 0.05, len(depths))
    psal = surface_psal - 1.5 * (1.0 - np.exp(-depths / 180.0)) + np.random.normal(0, 0.03, len(depths))

    return pd.DataFrame({
        "PRES": depths,
        "TEMP": np.round(temp, 2),
        "PSAL": np.round(psal, 2),
        "LATITUDE": [15.2] * len(depths),
        "LONGITUDE": [68.5] * len(depths)
    })

SYSTEM_PROMPT = """You are FloatChat AI, an expert, enthusiastic oceanographer and conversational assistant for global ARGO float data.
You can answer ANY question:
1. Greetings & Casual Chat: reply warmly and suggest ocean topics.
2. General Ocean Science: explain concepts clearly with markdown formatting.
3. ARGO Data & Profiling Requests: explain the depth dynamics and mixed layers, and at the VERY END output this exact trigger tag:
   [PLOT_DATA: region=REGION_NAME, depth=DEPTH_IN_METERS]

Do NOT output [PLOT_DATA] for general conversation or greetings."""

@app.post("/api/chat")
@app.post("/chat")
@app.post("/")
async def chat_api(req: ChatRequest):
    if not groq_client:
        return {"answer": "⚠️ Server error: GROQ_API_KEY environment variable is not configured in Vercel.", "is_data_query": False, "statistics": None, "chart_data": None}

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": req.message}
            ],
            temperature=0.3,
            max_tokens=1024
        )
        ai_text = response.choices[0].message.content
        plot_match = re.search(r'\[PLOT_DATA:\s*region=([^,]+),\s*depth=(\d+)\]', ai_text)

        if plot_match:
            region_name = plot_match.group(1).strip()
            max_depth = float(plot_match.group(2).strip())
            clean_answer = re.sub(r'\[PLOT_DATA:[^\]]+\]', '', ai_text).strip()
            df = generate_ocean_profile(region=region_name, max_depth=max_depth)
            
            stats = {
                "region": region_name,
                "records": len(df),
                "temp_min": float(df["TEMP"].min()),
                "temp_max": float(df["TEMP"].max()),
                "salinity_min": float(df["PSAL"].min()),
                "salinity_max": float(df["PSAL"].max()),
                "depth_range": [float(df["PRES"].min()), float(df["PRES"].max())]
            }

            return {
                "answer": clean_answer,
                "is_data_query": True,
                "statistics": stats,
                "chart_data": {"depth": df["PRES"].tolist(), "temp": df["TEMP"].tolist(), "psal": df["PSAL"].tolist()}
            }
        else:
            return {"answer": ai_text, "is_data_query": False, "statistics": None, "chart_data": None}
    except Exception as e:
        return {"answer": f"API Error: {str(e)}", "is_data_query": False, "statistics": None, "chart_data": None}
