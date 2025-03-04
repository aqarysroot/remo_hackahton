import os
import json
import base64
import requests
import openai
import asyncio
from fastapi import FastAPI, UploadFile, HTTPException, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from io import BytesIO
from PyPDF2 import PdfReader


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
async def submit_responses(responses: list[QuestionResponse]):
    async def process_response(response: QuestionResponse):
        try:
            audio_data = base64.b64decode(response.audioBlob)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error decoding audio for question {response.questionId}: {str(e)}"
            )

        # Transcribe audio from bytes
        transcript = await asyncio.to_thread(
            transcribe_audio_from_bytes, response.questionId, audio_data
        )
        transcript_text = transcript.get("text") if isinstance(transcript, dict) else transcript

        # Evaluate transcript using both the question and the answer
        evaluation_result = await asyncio.to_thread(
            evaluate_transcript, response.questionText, transcript_text
        )

        return {
            "questionId": response.questionId,
            "questionText": response.questionText,
            "answerText": transcript_text,
            "score": evaluation_result.get("evaluation", 50),
            "feedback": evaluation_result.get("comment", "No comment provided.")
        }

    results = await asyncio.gather(*(process_response(resp) for resp in responses))
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

def evaluate_transcript(question_text: str, answer_text: str) -> dict:
    """
    Sends the question and its transcribed answer to OpenAI for evaluation.
    The prompt instructs the model to evaluate how well the answer addresses the question
    on a scale of 1 to 100 and return a JSON object with keys:
      - evaluation: an integer score (1-100, where 1 is worst and 100 is best)
      - comment: a brief feedback comment.
    """
    prompt = (
        "Evaluate the following question and answer on a scale of 1 to 100, where 1 is the worst and 100 is the best. "
        "Consider how well the answer addresses the question, as well as clarity, correctness, and completeness. "
        "If the answer is exactly or almost identical to the question, assign a score of 80. "
        "Also provide a brief comment explaining your evaluation. "
        "Return the result as a valid JSON object with keys 'evaluation' and 'comment'.\n\n"
        f"Question:\n{question_text}\n\n"
        f"Answer:\n{answer_text}"
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


@app.get("/coding-question")
async def coding_question():
    """
    Returns a simple coding question related to JavaScript or React,
    along with sample input and sample output.
    """
    prompt = (
        "Give me a simple coding question related to JavaScript or React. "
        "Also provide sample input and sample output for the question if applicable. "
        "Return the result as a valid JSON object with keys 'question', 'input_samples', and 'output_samples'."
    )
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200,
        )
        response_text = completion.choices[0].message.content.strip()
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            # In case the response is not valid JSON, fall back to a simple text response.
            result = {
                "question": response_text,
                "input_samples": None,
                "output_samples": None
            }
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating coding question")


class CodingEvaluation(BaseModel):
    question: str
    answer: str


@app.post("/evaluate-answer")
async def evaluate_answer(evaluation: CodingEvaluation):
    """
    Accepts a coding question and answer, then evaluates the answer on a scale of 1 to 100.
    Returns a JSON object with 'evaluation' (score) and 'comment' (feedback).
    """
    result = await asyncio.to_thread(evaluate_coding_answer, evaluation.question, evaluation.answer)
    return JSONResponse(content=result)


def evaluate_coding_answer(question: str, answer: str) -> dict:
    """
    Evaluates a coding question and its answer on a scale of 1 to 100.
    Considers accuracy, clarity, and completeness.
    Returns a JSON object with keys 'evaluation' and 'comment'.
    """
    prompt = (
        "Evaluate the following coding question and its answer on a scale of 1 to 100, where 1 is the worst and 100 is the best. "
        "Consider how accurately, clearly, and completely the answer addresses the question. "
        "Provide a brief comment explaining your evaluation. "
        "Return the result as a valid JSON object with keys 'evaluation' and 'comment'.\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:\n{answer}"
    )
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a programming expert."},
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
        return {"evaluation": 50, "comment": "Could not evaluate answer."}

@app.post("/generate-interview-questions")
async def generate_interview_questions(
    job_description: str = Form(...),
    cv_file: UploadFile = File(...)
):
    """
    Accepts a job description as a form field and a candidate's CV as a file upload.
    If the file is a PDF, extracts the text using a PDF reader; otherwise, decodes the file.
    Then uses the job description and candidate CV to generate interview questions:
      - 3 technical questions that are specifically relevant to the job description.
      - 2 behavioral questions that assess soft skills related to the role.
    Each question is returned as an object with an id, text, type, and category.
    """
    try:
        # Check if the file is a PDF based on the content type or file extension.
        if cv_file.content_type == "application/pdf" or cv_file.filename.lower().endswith(".pdf"):
            cv_bytes = await cv_file.read()
            pdf_reader = PdfReader(BytesIO(cv_bytes))
            cv_text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    cv_text += page_text + "\n"
        else:
            # For non-PDF files, try to decode as text.
            cv_bytes = await cv_file.read()
            try:
                cv_text = cv_bytes.decode("utf-8")
            except UnicodeDecodeError:
                cv_text = cv_bytes.decode("latin-1")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CV file: {e}")

    # Construct prompts that explicitly require questions to be relevant to the job description.
    tech_prompt = (
        "Based on the following job description and candidate CV, generate 3 technical interview questions "
        "that assess the candidate's technical skills and overall fit for the role. "
        "IMPORTANT: The job description is for a Senior Python Developer. Ensure that the questions are "
        "specifically related to Python, relevant frameworks, and best practices expected at a senior level, "
        "and do not mention unrelated technologies.\n\n"
        f"Job Description:\n{job_description}\n\nCandidate CV:\n{cv_text}"
    )

    behav_prompt = (
        "Based on the following job description and candidate CV, generate 2 behavioral interview questions "
        "that assess the candidate's soft skills, teamwork, communication, and problem-solving abilities, "
        "relevant to the role of a Senior Python Developer. "
        "Return only the questions, each on a new line, with no numbering or additional commentary.\n\n"
        f"Job Description:\n{job_description}\n\nCandidate CV:\n{cv_text}"
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
    # Add technical questions.
    for i, line in enumerate(tech_lines, start=1):
        questions.append({
            "id": str(i),
            "text": line,
            "type": "technical",
            "category": "Technical"
        })
    # Add behavioral questions.
    for j, line in enumerate(behav_lines, start=len(tech_lines) + 1):
        questions.append({
            "id": str(j),
            "text": line,
            "type": "behavioral",
            "category": "Behavioral"
        })

    return JSONResponse(content=questions)


@app.post("/evaluate-cv")
async def evaluate_cv(
    job_description: str = Form(...),
    cv_file: UploadFile = File(...)
):
    """
    Accepts a job description as a form field and a candidate's CV as a file upload.
    If the file is a PDF, extracts the text using a PDF reader; otherwise, decodes the file.
    Processes the CV text and then evaluates how well it matches the job description.
    Returns a JSON object with 'evaluation' (score between 1 and 100) and 'comment' (feedback).
    """
    try:
        # Check if the file is a PDF based on the content type or file extension.
        if cv_file.content_type == "application/pdf" or cv_file.filename.lower().endswith(".pdf"):
            cv_bytes = await cv_file.read()
            pdf_reader = PdfReader(BytesIO(cv_bytes))
            cv_text = ""
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    cv_text += page_text + "\n"
        else:
            # For non-PDF files, try to decode as text
            cv_bytes = await cv_file.read()
            try:
                cv_text = cv_bytes.decode("utf-8")
            except UnicodeDecodeError:
                cv_text = cv_bytes.decode("latin-1")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CV file: {e}")

    # Process the CV text to extract relevant content
    processed_cv_text = await asyncio.to_thread(process_cv_text, cv_text)
    print("Processed CV Text:", processed_cv_text)

    # Evaluate the candidate's fit using the processed CV text
    result = await asyncio.to_thread(evaluate_candidate_cv, job_description, processed_cv_text)
    print("Evaluation Result:", result)
    return JSONResponse(content=result)


def process_cv_text(cv: str) -> str:
    """
    Processes the candidate's CV text to extract the relevant information.
    The assistant cleans up the CV by keeping details about education, experience, skills, and achievements.
    Returns the cleaned/summarized text.
    """
    prompt = (
        "Please extract and clean the relevant information from the following candidate CV. "
        "Focus on education, work experience, skills, and achievements. "
        "Return only the cleaned text without any extra commentary.\n\n"
        f"CV:\n{cv}"
    )
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that cleans and summarizes candidate CVs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=300,
        )
        result_text = completion.choices[0].message.content.strip()
        return result_text
    except Exception as e:
        print(f"Error processing CV text: {e}")
        # Fallback: return the original CV text if processing fails
        return cv


def evaluate_candidate_cv(job_description: str, cv: str) -> dict:
    """
    Uses the provided job description and candidate CV to evaluate the candidate's fit for the job.
    Returns a JSON object with keys 'evaluation' (score between 1 and 100) and 'comment' (feedback).
    """
    prompt = (
        "You are an experienced hiring manager evaluating a candidate's CV against a job description. "
        "Evaluate how well the candidate's CV fits the job description on a scale of 1 to 100, "
        "where 1 indicates a very poor fit and 100 indicates an excellent fit. "
        "Provide a brief comment explaining your evaluation. "
        "Return ONLY a valid JSON object with keys 'evaluation' and 'comment', with no additional text.\n\n"
        f"Job Description:\n{job_description}\n\n"
        f"Candidate CV:\n{cv}"
    )
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an experienced hiring manager."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=150,
        )
        response_text = completion.choices[0].message.content.strip()
        # Print raw response for debugging purposes
        print("Raw evaluation response:", response_text)
        if not response_text:
            print("Empty response received.")
            return {"evaluation": 50, "comment": "Could not evaluate the CV."}
        try:
            evaluation_data = json.loads(response_text)
        except json.JSONDecodeError as json_err:
            print("JSON decode error:", json_err, "Response:", response_text)
            # Attempt to extract a JSON substring from the response_text
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start != -1 and end != -1:
                try:
                    evaluation_data = json.loads(response_text[start:end])
                except Exception as e:
                    print("Fallback JSON extraction failed:", e)
                    evaluation_data = {"evaluation": 50, "comment": response_text}
            else:
                evaluation_data = {"evaluation": 50, "comment": response_text}
        return evaluation_data
    except Exception as e:
        print(f"Error evaluating CV: {e}")
        return {"evaluation": 50, "comment": "Could not evaluate the CV."}

# 1. Send in audio, and have it transcribed
# 2. We want to send it to chatgpt and get a response
# 3. We want to save the chat history to send back and forth for context.
