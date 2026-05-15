import os
import sys
import shutil
import subprocess
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Allowed audio formats
ALLOWED_EXTENSIONS = {'.mp3', '.wav', '.webm'}

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Lightweight endpoint to check if server is running."""
    return {"status": "online"}

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serves the frontend HTML file."""
    try:
        index_path = os.path.join(SCRIPT_DIR, "client.html")
        with open(index_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: client.html not found.</h1>"

@app.post("/predict")
def analyze_audio(file: UploadFile = File(...)):
    # Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            content={"status": "error", "message": f"Unsupported format. Allowed: mp3, wav, webm"},
            status_code=400
        )
    
    temp_input = os.path.join(SCRIPT_DIR, f"temp_in_{file.filename}")
    temp_output = os.path.join(SCRIPT_DIR, f"temp_out_{file.filename}.csv")
    demo_script = os.path.join(SCRIPT_DIR, "demo.py")
    
    try:
        with open(temp_input, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Use sys.executable for cross-platform Python path
        command = [
            sys.executable, "-u", demo_script, 
            "--audio_file", temp_input,
            "--output_file", temp_output,
            "--modality", "multimodal"
        ]

        print(f"Starting processing for {file.filename}...")
        
        # 2. MERGE stderr into stdout using stderr=subprocess.STDOUT
        with subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,  # <--- Merge streams
            text=True, 
            bufsize=1, 
            universal_newlines=True
        ) as p:
            # 3. Read and print in real-time
            for line in p.stdout:
                print(line, end='')  # Print exactly what comes in

        if p.returncode != 0:
            raise Exception("Script finished with errors.")

        if not os.path.exists(temp_output):
            raise Exception("Output CSV was not generated.")

        df = pd.read_csv(temp_output)
        results = df.to_dict(orient='records')
        csv_string = df.to_csv(index=False)

        return JSONResponse(content={
            "status": "success",
            "results": results,
            "csv_content": csv_string
        })

    except Exception as e:
        print(f"Server Error: {e}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
    
    finally:
        if os.path.exists(temp_input): os.remove(temp_input)
        if os.path.exists(temp_output): os.remove(temp_output)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)