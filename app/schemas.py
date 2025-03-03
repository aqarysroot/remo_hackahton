from uuid import UUID, uuid4
from typing import List, Optional
from pydantic import BaseModel, EmailStr


class Auth0CreateUser(BaseModel):
    user_id: UUID = None
    email: EmailStr
    password: str
    name: str
    username: str = str(uuid4())[:20]
    connection: str = 'Username-Password-Authentication'
    email_verified: bool = True


class Auth0Role(BaseModel):
    id: str = ''
    name: str
    description: str = ''


class InterviewQuestion(BaseModel):
    id: int
    question: str
    question_type: str
    expected_answer: Optional[str] = None
    candidate_answer: Optional[str] = None
    evaluation: Optional[str] = None


class InterviewSession(BaseModel):
    session_id: str = str(uuid4())
    job_description: str
    resume_text: str
    custom_questions: Optional[List[str]] = []
    question_count: int = 5
    question_types: List[str] = ["technical", "behavioral"]
    questions: Optional[List[InterviewQuestion]] = []
    current_question_index: int = 0
    completed: bool = False


class InterviewReport(BaseModel):
    session_id: str
    technical_evaluation: str
    behavioral_evaluation: str
    strengths: List[str]
    weaknesses: List[str]
    hire_recommendation: bool
    justification: str
