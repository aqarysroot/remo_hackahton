import os
import json
import base64
import requests
import openai
from fastapi import FastAPI, UploadFile, HTTPException, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPEN_AI_KEY")
# openai.organization = os.getenv("OPEN_AI_ORG")
elevenlabs_key = os.getenv("ELEVENLABS_KEY")

app = FastAPI()

origins = [
    "http://localhost:5174",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://localhost:3000",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


class QuestionResponse(BaseModel):
    questionId: str
    audioBlob: str  # Base64-encoded audio data


@app.post("/submit-responses")
async def submit_responses(responses: List[QuestionResponse]):
    results = []
    for response in responses:
        try:
            audio_data = base64.b64decode(response.audioBlob)
        except Exception as e:
            raise HTTPException(status_code=400,
                                detail=f"Error decoding audio for question {response.questionId}: {str(e)}")

        transcript = transcribe_audio_from_bytes(response.questionId, audio_data)
        results.append({
            "questionId": response.questionId,
            "transcript": transcript
        })
    return {"results": results}


@app.post("/talk")
async def post_audio(file: UploadFile):
    user_message = transcribe_audio(file)
    chat_response = get_chat_response(user_message)
    audio_output = text_to_speech(chat_response)

    def iterfile():
        yield audio_output

    return StreamingResponse(iterfile(), media_type="application/octet-stream")


@app.get("/clear")
async def clear_history():
    file = 'database.json'
    open(file, 'w')
    return {"message": "Chat history has been cleared"}


@app.get("/questions")
async def generate_questions():
    """
    Generates interview questions focused on React performance.

    The prompt instructs OpenAI to return 8 questions, one per line.
    The endpoint then iterates over each line, constructs an object for each question,
    and returns a list of these objects.
    """
    prompt = (
        "Generate 5 interview questions focused on React performance. "
        "Return only the questions, each on a new line, with no numbering or additional commentary."
    )

    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates interview questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300,
        )
        response_text = completion.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating questions: {e}")

    # Split the response into individual lines and remove empty lines
    lines = [line.strip() for line in response_text.split("\n") if line.strip()]

    # Build a list of question objects
    questions = []
    for i, line in enumerate(lines, start=1):
        questions.append({
            "id": str(i),
            "text": line,
            "type": "technical",
            "category": "Performance"
        })

    return questions


def transcribe_audio(file: UploadFile):
    # Save the blob first
    with open(file.filename, 'wb') as buffer:
        buffer.write(file.file.read())
    with open(file.filename, "rb") as audio_file:
        transcript = openai.Audio.transcribe(model="whisper-1", file=audio_file)
    os.remove(file.filename)
    print(transcript)
    return transcript


def transcribe_audio_from_bytes(identifier: str, audio_bytes: bytes) -> dict:
    """
    Writes binary audio data to a temporary file and transcribes it using OpenAI Whisper.
    """
    temp_filename = f"temp_{identifier}.wav"
    with open(temp_filename, "wb") as f:
        f.write(audio_bytes)
    with open(temp_filename, "rb") as audio_file:
        transcript = openai.Audio.transcribe(model="whisper-1", file=audio_file)
    os.remove(temp_filename)
    print(transcript)
    return transcript


def get_chat_response(user_message):
    messages = load_messages()
    messages.append({"role": "user", "content": user_message['text']})

    # Send to ChatGpt/OpenAi
    gpt_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    parsed_gpt_response = gpt_response['choices'][0]['message']['content']

    # Save messages
    save_messages(user_message['text'], parsed_gpt_response)

    return parsed_gpt_response


def load_messages():
    messages = []
    file = 'database.json'

    empty = os.stat(file).st_size == 0

    if not empty:
        with open(file) as db_file:
            data = json.load(db_file)
            for item in data:
                messages.append(item)
    else:
        messages.append(
            {"role": "system",
             "content": "You are interviewing the user for a front-end React developer position. Ask short questions that are relevant to a junior level developer. Your name is Greg. The user is Travis. Keep responses under 30 words and be funny sometimes."}
        )
    return messages


def save_messages(user_message, gpt_response):
    file = 'database.json'
    messages = load_messages()
    messages.append({"role": "user", "content": user_message})
    messages.append({"role": "assistant", "content": gpt_response})
    with open(file, 'w') as f:
        json.dump(messages, f)


def text_to_speech(text):
    voice_id = 'pNInz6obpgDQGcFmaJgB'

    body = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0,
            "similarity_boost": 0,
            "style": 0.5,
            "use_speaker_boost": True
        }
    }

    headers = {
        "Content-Type": "application/json",
        "accept": "audio/mpeg",
        "xi-api-key": elevenlabs_key
    }

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    try:
        response = requests.post(url, json=body, headers=headers)
        if response.status_code == 200:
            return response.content
        else:
            print('something went wrong')
    except Exception as e:
        print(e)

# 1. Send in audio, and have it transcribed
# 2. We want to send it to chatgpt and get a response
# 3. We want to save the chat history to send back and forth for context.
