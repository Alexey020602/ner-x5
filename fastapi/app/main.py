from fastapi import FastAPI
import os
from dotenv import load_dotenv
from app.model import initialize, predict

from app.models import NerRequest, NerResult

# script_directory = 
configuration = os.getenv("CONFIGURATION")

if configuration == "Development":
    load_dotenv(
        os.path.dirname(os.path.abspath(__file__)),
        os.path.pardir,
        ".env"
    )
    # script_directory = os.path.join(script_directory, os.path.pardir, os.path.pardir, "ModelIntegration")

# model_path = os.path.join(script_directory, 'model')
app = FastAPI()
model_repo_name = os.getenv("HuggingFace__ModelRepoName")
if not model_repo_name:
    raise ValueError("Environment not contains HuggingFace__ModelRepoName value")
model_repo_token = os.getenv("HuggingFace__ModelRepoToken")
if not model_repo_token:
    raise ValueError("Environment not contains HuggingFace__ModelRepoToken value")
initialize(model_repo_name, model_repo_token)
# model: NerModel = SpacyNerModel(model_path)

@app.post("/api/predict", response_model=list[NerResult])
async def root(request: NerRequest):
    result = predict(request.input)

    return [NerResult(start_index=record[0], end_index=record[1], entity=record[2]) for record in result]
