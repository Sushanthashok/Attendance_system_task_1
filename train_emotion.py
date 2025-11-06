import os, torch, torch.nn as nn, torch.optim as optim
from torchvision import models
from tqdm import tqdm
from utils import make_loaders

ROOT = 'data/emotions_prepared'
train_ld, val_ld, classes = make_loaders(ROOT, batch_size=64)
num_classes = len(classes)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

BEST = 0.0
for epoch in range(20):
    model.train()
    total, correct = 0, 0
    for x, y in tqdm(train_ld, desc=f'Epoch {epoch+1}/20 - train'):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        correct += (out.argmax(1)==y).sum().item()
        total += x.size(0)
    train_acc = correct/total
    scheduler.step(1.0-train_acc)
    if train_acc > BEST:
        BEST = train_acc
        os.makedirs('models', exist_ok=True)
        torch.save({'state_dict': model.state_dict(), 'classes': classes}, 'models/emotion_resnet18.pth')
    print(f'Epoch {epoch+1}: train_acc={train_acc:.3f}')
print('✅ Saved best Emotion model.')
