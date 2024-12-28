from fastapi import FastAPI, Form, Request
import numpy as np
import joblib
import os
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
# Initialize the FastAPI app
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


app = FastAPI()

# Ensure models exist before loading
models_dir = os.path.join(os.path.dirname(__file__), "./models")
college_model_path = "./models/college_model.pkl"
current_directory = os.getcwd()
print("Current directory:", current_directory)

if  not os.path.exists(college_model_path):
    raise FileNotFoundError("Model files not found. Please run 'models_creation.py' to generate them.")

# Load the pre-trained models
college_model = joblib.load(college_model_path)

##templates = Jinja2Templates(directory="templates")

_APP_GLOBAL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Use absolute path to templates directory
##templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates/"))

templates = Jinja2Templates(directory=os.path.join(_APP_GLOBAL_PATH, "templates/"))

print(f"tempaltepath {templates}")
@app.get("/", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict_admission")
async def predict_admission(payload: dict):
    academic_score = payload['academic_score']
    exam_score = payload['exam_score']
    extracurricular_score = payload['extracurricular_score']

    # Prepare the input data for prediction
    input_data = np.array([[academic_score, exam_score, extracurricular_score]])

    # Make predictions
    college_prediction = college_model.predict(input_data)[0]

    return {
        "college": college_prediction
    }

if __name__ == '__main__':
    import uvicorn

    uvicorn.run('main:app')
