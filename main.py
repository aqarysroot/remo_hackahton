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
    questionText: str
    audioBlob: str  # Base64-encoded audio data


@app.post("/submit-responses")
async def submit_responses(responses: List[QuestionResponse]):
    results = []
    for response in responses:
        try:
            audio_data = base64.b64decode(response.audioBlob)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error decoding audio for question {response.questionId}: {str(e)}"
            )
        transcript = transcribe_audio_from_bytes(response.questionId, audio_data)
        transcript_text = transcript.get("text") if isinstance(transcript, dict) else transcript

        evaluation_result = evaluate_transcript(transcript_text)
        results.append({
            "questionId": response.questionId,
            "questionText": response.questionText,
            "answerText": transcript_text,
            "score": evaluation_result.get("evaluation", 50),
            "feedback": evaluation_result.get("comment", "No comment provided.")
        })
    return {"results": results}


def transcribe_audio(file: UploadFile):
    """
    Saves the uploaded file temporarily, transcribes it using OpenAI Whisper,
    and then removes the temporary file.
    """
    try:
        with open(file.filename, 'wb') as buffer:
            buffer.write(file.file.read())
        with open(file.filename, "rb") as audio_file:
            transcript = openai.Audio.transcribe(model="whisper-1", file=audio_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        if os.path.exists(file.filename):
            os.remove(file.filename)
    print(transcript)
    return transcript


def transcribe_audio_from_bytes(identifier: str, audio_bytes: bytes) -> dict:
    """
    Writes binary audio data to a temporary file and transcribes it using OpenAI Whisper.
    """
    temp_filename = f"temp_{identifier}.wav"
    try:
        with open(temp_filename, "wb") as f:
            f.write(audio_bytes)
        with open(temp_filename, "rb") as audio_file:
            transcript = openai.Audio.transcribe(model="whisper-1", file=audio_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription from bytes failed: {e}")
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
    print(transcript)
    return transcript


def evaluate_transcript(transcript: str) -> dict:
    """
    Sends the transcript to OpenAI for evaluation.
    The prompt instructs the model to evaluate the transcript on a scale of 1 to 100
    and return a JSON object with keys:
      - evaluation: an integer score (1-100, where 1 is worst and 100 is best)
      - comment: a brief feedback comment.
    """
    prompt = (
        "Evaluate the following interview transcript on a scale of 1 to 100, where 1 is the worst and 100 is the best. "
        "Also provide a brief comment explaining your evaluation. Return the result as a JSON object with keys 'evaluation' and 'comment'.\n\n"
        f"Transcript:\n{transcript}"
    )
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert interviewer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150,
        )
        response_text = completion.choices[0].message.content.strip()
        evaluation_data = json.loads(response_text)
        return evaluation_data
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return {"evaluation": 50, "comment": "Could not evaluate transcript."}



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
    Generates interview questions in two parts:
      - 3 technical questions focused on React performance.
      - 2 behavioral questions focused on teamwork and communication.

    Each question is returned as an object with an id, text, type, and category.
    """
    tech_prompt = (
        "Generate 3 interview questions focused on React performance. "
        "Return only the questions, each on a new line, with no numbering or additional commentary."
    )

    try:
        tech_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates interview questions."},
                {"role": "user", "content": tech_prompt}
            ],
            temperature=0.7,
            max_tokens=300,
        )
        tech_response_text = tech_completion.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating technical questions: {e}")

    tech_lines = [line.strip() for line in tech_response_text.split("\n") if line.strip()]

    behav_prompt = (
        "Generate 2 behavioral interview questions focused on teamwork and communication in a software development environment. "
        "Return only the questions, each on a new line, with no numbering or additional commentary."
    )

    try:
        behav_completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates interview questions."},
                {"role": "user", "content": behav_prompt}
            ],
            temperature=0.7,
            max_tokens=300,
        )
        behav_response_text = behav_completion.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating behavioral questions: {e}")

    behav_lines = [line.strip() for line in behav_response_text.split("\n") if line.strip()]

    questions = []

    for j, line in enumerate(behav_lines, start=len(tech_lines) + 1):
        questions.append({
            "id": str(j),
            "text": line,
            "type": "behavioral",
            "category": "Behavioral"
        })

    for i, line in enumerate(tech_lines, start=1):
        questions.append({
            "id": str(i),
            "text": line,
            "type": "technical",
            "category": "Performance"
        })

    return questions



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
