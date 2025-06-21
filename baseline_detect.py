import sys
from transformers import pipeline


clf = pipeline("image-classification", model="prithivMLmods/Food-101-93M")


allergen_map = {
    "pad_thai": ["peanuts"],
    "donuts" : ["gluten"],
    "pizza": ["gluten"],
    "ice_cream": ["dairy"],
}

   
def predict_allergen(imagePath):
    predictions = clf(imagePath, top_k=5)
    allergens = set()
    for prediction in predictions:
        label = prediction['label']
        allergens.update(allergen_map.get(label, []))
    
    if allergens:
        print("Predicted allergens:", ", ".join(allergens))
    else:
        print("Predicted allergens: none")



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python baseline_detect.py <text>")
        sys.exit(1)
    else:
        imagePath = sys.argv[1]
        predict_allergen(imagePath)








