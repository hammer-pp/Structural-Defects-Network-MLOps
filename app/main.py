from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import torch
from torchvision import models, transforms
from PIL import Image
import io
import uuid, os
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader

load_dotenv()  # Load .env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

def upload_image_to_cloudinary(image_path: str, label: str):
    """
    Uploads image to Cloudinary under Users/{label}/<filename>
    """
    public_id = f"Users/{label}/{os.path.basename(image_path).split('.')[0]}"
    try:
        response = cloudinary.uploader.upload(
            image_path,
            public_id=public_id,
            overwrite=True,
            resource_type="image"
        )
        print("Feedback upload success to Cloudinary.")
        return response["secure_url"]
    except Exception as e:
        print("❌ Upload to Cloudinary failed:", e)
        return None

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # adjust class count
model.load_state_dict(torch.load("model/best_model.pth", map_location="cpu")) # test for the production unit only
model.eval()

# Preprocessing
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# Best applicable for ResNet-18 based models (custom dimension)
transform = transforms.Compose([
    transforms.Resize(256),              # Resize shorter side to 256 px
    transforms.CenterCrop(224),          # Crop center to 224×224
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

MODEL_DIR = "model"
AVAILABLE_MODELS = {
    "resnet": os.path.join(MODEL_DIR, "resnet_model.pth"),
    "mobilenet": os.path.join(MODEL_DIR, "mobilenet_model.pth"),
    "best": os.path.join(MODEL_DIR, "best_model.pth")
}

MODEL_LABELS = ["Non-cracked", "Cracked"]

def load_model(model_name: str):
    if model_name == "resnet":
        net = models.resnet18()
        net.fc = torch.nn.Linear(net.fc.in_features, 2)
    elif model_name == "mobilenet":
        net = models.mobilenet_v3_large()
        net.classifier[1] = torch.nn.Linear(net.fc.in_features, 2)
    elif model_name == "best":
        net = models.resnet18()
        net.fc = torch.nn.Linear(net.fc.in_features, 2)
    else:
        raise ValueError("Unknown model")

    state_path = AVAILABLE_MODELS[model_name]
    net.load_state_dict(torch.load(state_path, map_location=device))
    net.to(device)
    net.eval()
    return net


labels = ["Non-cracked", "Cracked"]  # adjust to match your training (Thunder's notebook)

@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...), model_name: str = Form(...)):
    model = load_model(model_name)
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Generate UUID-based filename
    unique_id = uuid.uuid4().hex
    ext = os.path.splitext(file.filename)[-1]
    filename = f"{unique_id}{ext}"

    # Save to temp path
    temp_path = f"static/test/temp/{filename}"
    os.makedirs("static/test/temp", exist_ok=True)
    with open(temp_path, "wb") as buffer:
        buffer.write(contents)

    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
        prediction = labels[pred.item()]

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": prediction,
        "filename": filename,
        "image_url": f"/{temp_path}",
        "selected_model": model_name
    })

@app.post("/feedback", response_class=HTMLResponse)
async def feedback(
    request: Request,
    filename: str = Form(...),
    predicted_label: str = Form(...),
    feedback: str = Form(...)
):
    src_path = f"static/test/temp/{filename}"

    if feedback == "correct":
        dst_label = predicted_label
    else:
        # Invert the binary label
        dst_label = "no_crack" if predicted_label == "crack" else "crack"

    dst_dir = f"static/test/confirmed/{dst_label}"

    # For local marking 
    # os.makedirs(dst_dir, exist_ok=True)
    # dst_path = os.path.join(dst_dir, filename)
    # os.rename(src_path, dst_path)
    
    # Upload to Cloudinary
    cloudinary_url = upload_image_to_cloudinary(src_path, dst_label)

    # Remove temp file after upload
    os.remove(src_path)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": f"Feedback recorded: {feedback.upper()}",
        # "image_url": f"/{dst_path}"
        "image_url" : cloudinary_url
    })