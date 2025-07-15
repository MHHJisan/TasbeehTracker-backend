import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
import google.api_core.exceptions
import time

app = FastAPI()
load_dotenv()

# Configure Gemini API
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("⚠️ Warning: GOOGLE_API_KEY not found in environment variables")
    api_key = "dummy_key"
genai.configure(api_key=api_key)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        # Read uploaded file contents
        contents = await file.read()
        print(f"Received audio file size: {len(contents)} bytes")

        # Create a temporary file and keep it open until transcription is done
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_path = temp_file.name
        temp_file.write(contents)
        temp_file.flush()  # Ensure data is written to disk
        temp_file.close()  # Close the file to allow reading by Gemini API
        print(f"Saved uploaded audio to: {temp_path}")

        max_retries = 3
        retry_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                # Verify file exists before processing
                if not os.path.exists(temp_path):
                    raise FileNotFoundError(f"Temporary file not found: {temp_path}")

                # Load Gemini model
                model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
                with open(temp_path, "rb") as audio_file:
                    audio_data = audio_file.read()
                    print(f"Read audio file size: {len(audio_data)} bytes")
                    response = model.generate_content(
                        [{"mime_type": "audio/wav", "data": audio_data}],
                        stream=False,
                    )

                transcription = response.text.strip()
                if transcription:
                    print("✅ Transcription:", transcription)
                    return {"transcription": transcription}
                else:
                    print("⚠️ Invalid transcription result:", response)
                    return {"error": "Invalid transcription result"}

            except google.api_core.exceptions.ResourceExhausted as e:
                print(f"❌ Quota exceeded, attempt {attempt + 1}/{max_retries}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    continue
                return {"error": "Quota exceeded. Please try again later or check your plan."}
            except Exception as e:
                print("❌ Error:", str(e))
                return {"error": str(e)}

    except Exception as e:
        print("❌ Upload error:", str(e))
        return {"error": f"Failed to process audio: {str(e)}"}
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                print(f"Deleted temporary file: {temp_path}")
        except Exception as e:
            print(f"Failed to delete temp file {temp_path}: {str(e)}")