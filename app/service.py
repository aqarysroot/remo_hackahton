import openai
import os
import json
import requests
import uuid
from typing import List, Optional

from app.settings import settings
from app.schemas import InterviewSession, InterviewQuestion, InterviewReport


def transcribe_audio(file):
    # Save the blob first
    with open(file.filename, 'wb') as buffer:
        buffer.write(file.file.read())
    audio_file = open(file.filename, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    print(transcript)
    os.remove(file.filename)  # Clean up the file after transcription
    return transcript


def get_chat_response(user_message):
    messages = load_messages()
    messages.append({"role": "user", "content": user_message['text']})

    # Send to ChatGPT/OpenAI
    gpt_response = gpt_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    parsed_gpt_response = gpt_response['choices'][0]['message']['content']

    # Save messages
    save_messages(user_message['text'], parsed_gpt_response)

    return parsed_gpt_response


def extract_resume_text(resume_file):
    """Extract text from a resume file (PDF or DOCX)"""
    # For simplicity, we'll just read the file as text
    # In a real implementation, you'd want to use libraries like PyPDF2 or python-docx
    content = resume_file.file.read().decode('utf-8', errors='ignore')
    return content


def generate_interview_questions(session: InterviewSession) -> List[InterviewQuestion]:
    """Generate interview questions based on job description and resume"""
    
    # Construct the prompt for GPT
    prompt = f"""
    You are an expert technical interviewer. Create {session.question_count} interview questions based on the following:
    
    JOB DESCRIPTION:
    {session.job_description}
    
    CANDIDATE RESUME:
    {session.resume_text}
    
    QUESTION TYPES NEEDED: {', '.join(session.question_types)}
    
    For each question, provide:
    1. The question text
    2. The question type (technical or behavioral)
    3. What would constitute a good answer
    
    Format your response as a JSON array of objects with the following structure:
    [
      {
        "question": "Question text here",
        "question_type": "technical or behavioral",
        "expected_answer": "Description of what constitutes a good answer"
      }
    ]
    
    If there are custom questions that must be included, here they are:
    {session.custom_questions}
    """
    
    # Call GPT to generate questions
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert technical interviewer who creates tailored interview questions."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    # Parse the response
    try:
        content = response['choices'][0]['message']['content']
        # Extract JSON from the response (it might be wrapped in markdown code blocks)
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].strip()
        else:
            json_str = content.strip()
            
        questions_data = json.loads(json_str)
        
        # Convert to InterviewQuestion objects
        questions = []
        for i, q_data in enumerate(questions_data):
            questions.append(InterviewQuestion(
                id=i,
                question=q_data["question"],
                question_type=q_data["question_type"],
                expected_answer=q_data.get("expected_answer", "")
            ))
        
        return questions
    except Exception as e:
        print(f"Error parsing questions: {e}")
        # Fallback to a simple question if parsing fails
        return [InterviewQuestion(
            id=0,
            question="Tell me about your experience with the technologies mentioned in the job description.",
            question_type="general",
            expected_answer="A comprehensive overview of relevant experience."
        )]


def evaluate_interview_response(question: InterviewQuestion, response: str) -> str:
    """Evaluate a candidate's response to an interview question"""
    
    prompt = f"""
    Evaluate the candidate's response to this interview question:
    
    QUESTION: {question.question}
    QUESTION TYPE: {question.question_type}
    EXPECTED GOOD ANSWER WOULD INCLUDE: {question.expected_answer}
    
    CANDIDATE'S RESPONSE: {response}
    
    Provide a brief evaluation of the response quality (1-3 sentences).
    """
    
    evaluation_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert at evaluating interview responses."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    return evaluation_response['choices'][0]['message']['content']


def save_interview_session(session: InterviewSession):
    """Save interview session to a JSON file"""
    sessions_dir = 'interview_sessions'
    os.makedirs(sessions_dir, exist_ok=True)
    
    file_path = os.path.join(sessions_dir, f"{session.session_id}.json")
    with open(file_path, 'w') as f:
        f.write(session.json())
    
    return session.session_id


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
             "content": "You are an AI interviewer conducting a technical interview. Ask relevant questions based on the job description and candidate's resume. Keep responses professional and concise."}
        )
    return messages


def save_messages(user_message, gpt_response):
    file = 'database.json'
    messages = load_messages()
    messages.append({"role": "user", "content": user_message})
    messages.append({"role": "assistant", "content": gpt_response})
    with open(file, 'w') as f:
        json.dump(messages, f)


def load_interview_session(session_id: str) -> Optional[InterviewSession]:
    """Load an interview session from a JSON file"""
    file_path = os.path.join('interview_sessions', f"{session_id}.json")
    
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r') as f:
        session_data = json.load(f)
    
    return InterviewSession.parse_obj(session_data)


def update_interview_session(session: InterviewSession):
    """Update an existing interview session"""
    save_interview_session(session)


def generate_interview_report(session_id: str) -> InterviewReport:
    """Generate a final report for an interview session"""
    session = load_interview_session(session_id)
    
    if not session or not session.questions:
        raise ValueError("Interview session not found or no questions available")
    
    # Prepare data for the report
    questions_and_answers = []
    for q in session.questions:
        if q.candidate_answer:
            questions_and_answers.append({
                "question": q.question,
                "type": q.question_type,
                "answer": q.candidate_answer,
                "evaluation": q.evaluation
            })
    
    # Generate the report using GPT
    prompt = f"""
    Generate a comprehensive interview report based on the following interview data:
    
    JOB DESCRIPTION:
    {session.job_description}
    
    CANDIDATE RESUME:
    {session.resume_text}
    
    INTERVIEW QUESTIONS AND ANSWERS:
    {json.dumps(questions_and_answers, indent=2)}
    
    Please provide:
    1. Technical evaluation (2-3 paragraphs)
    2. Behavioral evaluation (1-2 paragraphs)
    3. Key strengths (3-5 bullet points)
    4. Areas for improvement (3-5 bullet points)
    5. Hire recommendation (Yes/No)
    6. Justification for the recommendation (2-3 sentences)
    
    Format your response as JSON with the following structure:
    {
      "technical_evaluation": "text here",
      "behavioral_evaluation": "text here",
      "strengths": ["strength 1", "strength 2", ...],
      "weaknesses": ["weakness 1", "weakness 2", ...],
      "hire_recommendation": true or false,
      "justification": "text here"
    }
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an expert at evaluating technical interviews and providing hiring recommendations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    
    # Parse the response
    try:
        content = response['choices'][0]['message']['content']
        # Extract JSON from the response
        if "```json" in content:
            json_str = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            json_str = content.split("```")[1].strip()
        else:
            json_str = content.strip()
            
        report_data = json.loads(json_str)
        
        # Create and return the report
        return InterviewReport(
            session_id=session_id,
            technical_evaluation=report_data["technical_evaluation"],
            behavioral_evaluation=report_data["behavioral_evaluation"],
            strengths=report_data["strengths"],
            weaknesses=report_data["weaknesses"],
            hire_recommendation=report_data["hire_recommendation"],
            justification=report_data["justification"]
        )
    except Exception as e:
        print(f"Error generating report: {e}")
        raise


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
        "xi-api-key": settings.elevenlabs_key
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


def get_next_question(session_id: str) -> Optional[InterviewQuestion]:
    """Get the next question for an interview session"""
    session = load_interview_session(session_id)
    
    if not session or not session.questions:
        return None
    
    if session.current_question_index >= len(session.questions):
        session.completed = True
        update_interview_session(session)
        return None
    
    return session.questions[session.current_question_index]


def save_answer_and_evaluate(session_id: str, answer: str) -> Optional[str]:
    """Save a candidate's answer and evaluate it"""
    session = load_interview_session(session_id)
    
    if not session or session.completed:
        return None
    
    current_q_index = session.current_question_index
    if current_q_index >= len(session.questions):
        return None
    
    # Save the answer
    session.questions[current_q_index].candidate_answer = answer
    
    # Evaluate the answer
    evaluation = evaluate_interview_response(session.questions[current_q_index], answer)
    session.questions[current_q_index].evaluation = evaluation
    
    # Move to the next question
    session.current_question_index += 1
    update_interview_session(session)
    
    return evaluation
