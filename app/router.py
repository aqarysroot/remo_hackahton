from typing import Optional, List
from fastapi import UploadFile, Form, File, HTTPException, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.boot import app
from app.service import (
    transcribe_audio, get_chat_response, text_to_speech,
    extract_resume_text, generate_interview_questions, save_interview_session,
    load_interview_session, get_next_question, save_answer_and_evaluate,
    generate_interview_report
)
from app.schemas import InterviewSession, InterviewQuestion, InterviewReport

class QuestionRequest(BaseModel):
    job_description: str
    question_count: int = 5
    question_types: List[str] = ["technical", "behavioral"]
    custom_questions: List[str] = []

@app.post("/questions")
async def generate_questions(
    job_description: str = Form(...),
    question_count: int = Form(5),
    question_types: str = Form("technical,behavioral"),
    custom_questions: str = Form(""),
    resume: UploadFile = File(...)
):
    """Generate interview questions based on job description and resume"""
    try:
        # Extract text from resume
        resume_text = extract_resume_text(resume)
        
        # Parse question types and custom questions
        question_types_list = [qt.strip() for qt in question_types.split(",")]
        custom_questions_list = [q.strip() for q in custom_questions.split("\n") if q.strip()]
        
        # Create interview session
        session = InterviewSession(
            job_description=job_description,
            resume_text=resume_text,
            question_count=question_count,
            question_types=question_types_list,
            custom_questions=custom_questions_list
        )
        
        # Generate questions
        questions = generate_interview_questions(session)
        session.questions = questions
        
        # Save the session
        session_id = save_interview_session(session)
        
        # Return the session ID and questions
        return {
            "session_id": session_id,
            "questions": [q.dict() for q in questions]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating questions: {str(e)}")

@app.get("/interview/{session_id}/next-question")
async def next_question(session_id: str):
    """Get the next question for an interview session"""
    question = get_next_question(session_id)
    if not question:
        return {"completed": True}
    
    return {"completed": False, "question": question.dict()}

@app.post("/interview/{session_id}/answer")
async def submit_answer(session_id: str, answer: str = Body(..., embed=True)):
    """Submit an answer to the current question"""
    evaluation = save_answer_and_evaluate(session_id, answer)
    if not evaluation:
        raise HTTPException(status_code=404, detail="Session not found or completed")
    
    return {"evaluation": evaluation}

@app.get("/interview/{session_id}/report")
async def get_report(session_id: str):
    """Generate a final report for the interview"""
    try:
        report = generate_interview_report(session_id)
        return report.dict()
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")

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
    with open('database.json', 'w') as f:
        f.write('')
    return {"message": "Chat history has been cleared"}
