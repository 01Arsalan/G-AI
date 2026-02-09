from fastapi import FastAPI, Path, Query
from typing import Optional
from pydantic import BaseModel

app = FastAPI()


students = {
    1: {"name": "Alice", "age": 20},
    2: {"name": "Bob", "age": 22},
}

teachers = {
    1: {"name": "Mr. Smith", "subject": "Math"},
    2: {"name": "Ms. Johnson", "subject": "English"},
}

class studentModel(BaseModel):
    name: str
    age: int

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/student-id/{student_id}")
def get_student(student_id: Optional[int]= Path(..., description="The ID of the student to retrieve", ge = 1, le = 3)):
    student = students.get(student_id)
    if student:
        return student
    else:
        return {"Not found, student_id:": student_id}
    
@app.get("/student-name")
def get_student(
    student_name: Optional[str] = Query(
        None, 
        description="The NAME of the student to retrieve"
    )
):
    if student_name is None:
        return {"error": "Please provide a student name as a query parameter"}

    for student_id, student in students.items():
        if student["name"].lower() == student_name.lower():
            return student

    return {"error": "Not found", "student_name": student_name}

@app.get("/teachers/{teacher_id}")
def get_teacher(teacher_id: int, q: Optional[str] = None):
    teacher = teachers.get(teacher_id)
    if teacher:
        if q:
            return {"teacher": teacher, "query": q}
        return teacher
    else:
        return {"Not found, teacher_id:": teacher_id}
    

@app.post("/create-student/{student_id}")
def create_student(student_id: int, student: studentModel, gender: Optional[str] = Query(None, description="Gender of the student")):
    print(student)
    for s in students.values():
        if s["name"] == student.name:
            return {"error": "Student already exists."}
    students[student_id] = student.model_dump()
    return {"message": "Student created successfully.", "student": students[student_id]}






# uvicorn fast_api_uvcorn:app --reload