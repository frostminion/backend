from fastapi import FastAPI
from pydantic import BaseModel
from chatbot_popu import load_popu_chain
from chatbot_eco import load_eco_chain

app = FastAPI()

class Query(BaseModel):
    question: str

@app.on_event("startup")
def startup_event():
    global popu_chain, eco_chain
    popu_chain = load_popu_chain()
    eco_chain = load_eco_chain()

@app.post("/chat/popu")
def chat_popu(q: Query):
    return {"result": popu_chain.run(q.question)}

@app.post("/chat/eco")
def chat_eco(q: Query):
    return {"result": eco_chain.run(q.question)}
