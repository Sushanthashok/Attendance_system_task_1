import os, glob, random, shutil
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN

SRC_STUDENTS = 'data/students_raw'
DST_STUDENTS = 'data/students_cropped'
SRC_EMO      = 'data/emotions_raw'
DST_EMO      = 'data/emotions_prepared'

mtcnn = MTCNN(image_size=224, margin=20)

os.makedirs(DST_STUDENTS, exist_ok=True)
os.makedirs(DST_EMO, exist_ok=True)

# Crop student faces
for person in os.listdir(SRC_STUDENTS):
    src_dir = os.path.join(SRC_STUDENTS, person)
    if not os.path.isdir(src_dir): continue
    dst_dir = os.path.join(DST_STUDENTS, person)
    os.makedirs(dst_dir, exist_ok=True)
    for imgp in tqdm(glob.glob(os.path.join(src_dir, '*'))):
        try:
            img = Image.open(imgp).convert('RGB')
            face = mtcnn(img)
            if face is not None:
                out = Image.fromarray(face.permute(1,2,0).byte().numpy())
                out.save(os.path.join(dst_dir, os.path.basename(imgp)))
        except Exception:
            pass

# Split emotion dataset
random.seed(42)
classes = [d for d in os.listdir(SRC_EMO) if os.path.isdir(os.path.join(SRC_EMO, d))]
for cls in classes:
    imgs = glob.glob(os.path.join(SRC_EMO, cls, '*'))
    random.shuffle(imgs)
    split = int(0.85*len(imgs))
    train_imgs, val_imgs = imgs[:split], imgs[split:]
    for split_name, split_imgs in [('train', train_imgs), ('val', val_imgs)]:
        out_dir = os.path.join(DST_EMO, split_name, cls)
        os.makedirs(out_dir, exist_ok=True)
        for imgp in tqdm(split_imgs, desc=f'{cls}-{split_name}'):
            try:
                img = Image.open(imgp).convert('RGB')
                face = mtcnn(img)
                if face is not None:
                    out = Image.fromarray(face.permute(1,2,0).byte().numpy())
                    out.save(os.path.join(out_dir, os.path.basename(imgp)))
            except Exception:
                pass

print('✅ Finished preparing datasets.')
