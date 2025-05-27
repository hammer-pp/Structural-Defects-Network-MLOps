from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import torch
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # adjust class count
model.load_state_dict(torch.load("model/prod.pth", map_location="cpu")) # test for the production unit only
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


labels = ["Non-cracked", "Cracked"]  # adjust to match your training (Thunder's notebook)

@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        prediction = labels[pred.item()]

    return templates.TemplateResponse("index.html", {"request": request, "result": prediction})
