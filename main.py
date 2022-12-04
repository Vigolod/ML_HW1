from fastapi import FastAPI, File, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
from parsing import medians, model
from parsing import prepare_df

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

@app.post("/predict_item")
async def predict_item(item: Item) -> float:
    item_pd = prepare_df(pd.DataFrame(dict(item), index=[0])).drop('selling_price', axis=1)
    result = model.predict(item_pd)[0]
    return result

@app.post("/prefict_items", response_class=FileResponse)
async def predict_item(file: UploadFile = File(...)):
    data = pd.read_csv(file.file, index_col=0)
    if 'selling_price' in data.columns:
        data.drop('selling_price', axis=1, inplace=True)
    df = prepare_df(data)
    y = model.predict(df)
    data['selling_price'] = y
    data.to_csv('response.csv')
    return FileResponse('response.csv')