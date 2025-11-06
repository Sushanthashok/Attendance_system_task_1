import os, datetime
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMG_SIZE = 224

train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def make_loaders(root, batch_size=32):
    train_dir = os.path.join(root, 'train')
    val_dir   = os.path.join(root, 'val')
    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds   = datasets.ImageFolder(val_dir,   transform=val_tfms)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_ld   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)


    return train_ld, val_ld, train_ds.classes

ASIA_KOLKATA_UTC_OFFSET_MIN = 330

def within_window(start_h=9, start_m=30, end_h=10, end_m=0):
    """Return True if local time (Asia/Kolkata) is within the window."""
    now = datetime.datetime.utcnow()
    ist = now + datetime.timedelta(minutes=ASIA_KOLKATA_UTC_OFFSET_MIN)
    start = ist.replace(hour=start_h, minute=start_m, second=0, microsecond=0)
    end   = ist.replace(hour=end_h,   minute=end_m,   second=0, microsecond=0)
    return start <= ist <= end
