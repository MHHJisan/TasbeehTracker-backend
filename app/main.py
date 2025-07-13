import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai

# Initialize FastAPI app
app = FastAPI()

# Load environment variables from .env
load_dotenv()

# Configure Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set up CORS (open for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with allowed domains in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Endpoint 1: Only uploads audio (useful for testing upload separately)
@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        with open("uploaded_audio.wav", "wb") as f:
            f.write(contents)
        return {"status": "uploaded", "filename": file.filename}
    except Exception as e:
        return {"error": f"Upload failed: {str(e)}"}

# 🔹 Endpoint 2: Upload + transcribe using Gemini
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    contents = await file.read()
    print(f"Received audio file size: {len(contents)} bytes")


    # Save uploaded audio temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
        print(f"Saved uploaded audio to: {tmp_path}")

    try:
        # Load Gemini model
        model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

        # Read audio and send to Gemini
        with open(tmp_path, "rb") as audio_file:
            audio_data = audio_file.read()
            print(f"Read audio file size: {len(audio_data)} bytes")
            response = model.generate_content(
                [genai.types.ContentPart.from_data(data=audio_file.read(), mime_type="audio/wav")],
                stream=False,
            )

        transcription = response.text.strip() 
        

        if transcription:
            print("✅ Transcription:", transcription)
            return {"transcription": transcription}
        else:
            print("⚠️ Invalid transcription result:", response)
            return {"error": "Invalid transcription result"}

    except Exception as e:
        print("❌ Error:", e)
        return {"error": str(e)}
