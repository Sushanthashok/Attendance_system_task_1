# train_faceid.py
import os, torch, torch.nn as nn, torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

ROOT = 'data/students_cropped'
IMG_SIZE = 224

tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.02),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

full_ds   = datasets.ImageFolder(ROOT, transform=tfms)
classes   = full_ds.classes
n_total   = len(full_ds)
n_train   = int(0.85*n_total)
n_val     = n_total - n_train
train_ds, val_ds = random_split(full_ds, [n_train, n_val])

# validation should not augment
val_ds.dataset.transform = val_tfms

train_ld = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
val_ld   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)



num_classes = len(classes)

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

BEST = 0.0
for epoch in range(15):
    model.train()
    total, correct = 0, 0
    for x, y in tqdm(train_ld, desc=f'Epoch {epoch+1}/15 - train'):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(); out = model(x)
        loss = criterion(out, y); loss.backward(); optimizer.step()
        correct += (out.argmax(1)==y).sum().item(); total += x.size(0)
    train_acc = correct/total if total else 0.0

    # val
    model.eval(); vtotal, vcorrect = 0, 0
    with torch.no_grad():
        for x, y in val_ld:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            vcorrect += (pred==y).sum().item(); vtotal += x.size(0)
    val_acc = vcorrect/vtotal if vtotal else train_acc
    scheduler.step(1.0 - val_acc)

    if val_acc > BEST:
        BEST = val_acc
        os.makedirs('models', exist_ok=True)
        torch.save({'state_dict': model.state_dict(), 'classes': classes}, 'models/faceid_mobilenetv2.pth')
    print(f'Epoch {epoch+1}: train={train_acc:.3f} val={val_acc:.3f}')
print('✅ Saved best FaceID model.')
