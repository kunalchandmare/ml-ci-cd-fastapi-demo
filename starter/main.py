# Import Union since our Item object will have tags that can be strings or a list.
from typing import Union

from fastapi import FastAPI
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel

# Declare the data object with its components and their type.
class TaggedItem(BaseModel):
    name: str
    tags: Union[str, list]
    item_id: int

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello! This is API for to connect our Model which is \n"
                       " a supervised binary classification model to predict whether an adult earns more than $50,000\n "
                       "per year based on demographic and employment features from the 1994 US Census data."}

@app.post("/items/")
async def create_item(item: TaggedItem):
    return item