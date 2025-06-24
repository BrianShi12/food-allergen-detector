import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import Food101
import torchvision.models as models
from torchvision.models import ResNet34_Weights
import torch.nn as nn
import torch.optim as optim
from collections import Counter




def make_allergen_target(food_label, classes, allergen_classes, num_allergens, device):
    target = torch.zeros(num_allergens, dtype=torch.float32)
    food_name = classes[food_label]
    target[allergen_classes[food_name]] = 1.0
    return target


def main(): 
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    EPOCHS     = 5
    LR         = 1e-4



    allergen_classes = {
        "sushi": 0,
        "ice_cream" : 1,
        "cheesecake" : 1,
        "pizza" :2,
        "bagel": 2
    }

    num_allergens = 3


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    train_ds = Food101(root="data", split="train", download=True, transform=transform)
    test_ds = Food101(root="data", split="test", download=True, transform=transform)

    wanted = set(allergen_classes.keys())


    def filter_ds(ds):
        indices = [i for i, (_, label) in enumerate(ds) if ds.classes[label] in wanted]
        return Subset(ds, indices)

    train_sub = filter_ds(train_ds)
    val_sub = filter_ds(test_ds)


    counter = Counter()
    for _, lbl in train_sub:
        food = train_ds.classes[lbl]
        counter[food] += 1

    print("Training counts by food class:")
    for cls, cnt in counter.items():
        print(f"  {cls:15s}: {cnt}")

    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_sub,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)



    model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, num_allergens)
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        run_loss = 0
        for images, food_lbls in train_loader:
            images = images.to(DEVICE)
            targets = torch.stack([
                make_allergen_target(lbl.item(), train_ds.classes, allergen_classes, num_allergens, DEVICE)
                for lbl in food_lbls
            ]).to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * images.size(0)

        print(f"Epoch {epoch+1}/{EPOCHS} — train loss: {run_loss/len(train_sub):.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, food_lbls in val_loader:
                images = images.to(DEVICE)
                targets = torch.stack([
                    make_allergen_target(lbl.item(), train_ds.classes, allergen_classes, num_allergens, DEVICE)
                    for lbl in food_lbls
                ]).to(DEVICE)

                outputs = model(images)
                val_loss += criterion(outputs, targets).item() * images.size(0)

        print(f"          — val loss:   {val_loss/len(val_sub):.4f}\n")

    torch.save(model.state_dict(), "allergen_model.pth")
    print("Saved allergen_model.pth")


if __name__ == "__main__":
    main()