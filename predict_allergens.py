import sys
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# 1) Configuration
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH    = "allergen_model.pth"
THRESHOLD     = 0.5   # tune this if you need fewer/more false positives

# 2) Allergen labels by index
idx2allergen = {
    0: "peanuts",
    1: "dairy",
    2: "gluten",
}

# 3) Image preprocessing (must match your training transforms)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # If you used ImageNet normalization during training add:
    # transforms.Normalize([0.485, 0.456, 0.406],
    #                      [0.229, 0.224, 0.225])
])

# 4) Build the model architecture & load weights
model = models.resnet34(pretrained=False)
in_f  = model.fc.in_features
model.fc = nn.Linear(in_f, len(idx2allergen))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

def predict(image_path):
    # a) Load & preprocess
    img = Image.open(image_path).convert("RGB")
    x   = transform(img).unsqueeze(0).to(DEVICE)  # shape [1,3,224,224]

    # b) Inference
    with torch.no_grad():
        logits = model(x)                         # [1,3]
        probs  = torch.sigmoid(logits)[0]         # [3]

    # c) Threshold & collect
    detected = [idx2allergen[i] for i,p in enumerate(probs) if p >= THRESHOLD]

    # d) Print results
    print(f"\nAllergen probabilities:")
    for i, p in enumerate(probs):
        print(f"  {idx2allergen[i]:<8}: {p.item():.2%}")
    if detected:
        print("\nPredicted allergen(s):", ", ".join(detected))
    else:
        print("\nPredicted allergen(s): none")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_allergens.py path/to/image.jpg")
        sys.exit(1)
    predict(sys.argv[1])
