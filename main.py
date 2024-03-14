from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from kd import trainAndEvaluate

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def f(
        num_train_epochs: float,
        fp16: bool,
        logging_strategy: str,
        evaluation_strategy: str,
        save_strategy: str,
        load_best_model_at_end: bool,
        temperature: float,
        lambda_param: float
):
    return EventSourceResponse(
        trainAndEvaluate(
            num_train_epochs,
            fp16,
            logging_strategy,
            evaluation_strategy,
            save_strategy,
            load_best_model_at_end,
            temperature,
            lambda_param
        )
    )
