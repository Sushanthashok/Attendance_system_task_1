# run_attendance.py
import os, datetime, csv
import cv2
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms, models

from utils import within_window  # uses IST window 09:30–10:00

# ---- config ----
IMG_SIZE = 224
ATTN_START = (9, 30)  # 09:30
ATTN_END   = (10, 0)  # 10:00
CAM_INDEX  = 0        # change to 1 if external webcam

# preprocessing for both models
preproc = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ---- load models ----
# FaceID
face_pkg = torch.load('models/faceid_mobilenetv2.pth', map_location='cpu')
face_classes = face_pkg['classes']   # e.g., ['101_AkshayKumar', '102_AliaBhatt', ...]
face_model = models.mobilenet_v2(weights=None)
face_model.classifier[1] = torch.nn.Linear(face_model.last_channel, len(face_classes))
face_model.load_state_dict(face_pkg['state_dict'], strict=True)
face_model.eval()

# Emotion
emo_pkg = torch.load('models/emotion_resnet18.pth', map_location='cpu')
emo_classes = emo_pkg['classes']     # e.g., ['angry','disgust','fear','happy','neutral','sad','surprise']
emo_model = models.resnet18(weights=None)
emo_model.fc = torch.nn.Linear(emo_model.fc.in_features, len(emo_classes))
emo_model.load_state_dict(emo_pkg['state_dict'], strict=True)
emo_model.eval()

# Face detector
mtcnn = MTCNN(image_size=IMG_SIZE, margin=20)

# ---- read roster ----
roster = []
with open('roster.csv', 'r', newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        roster.append((row['roll_no'], row['name']))
rollnos = {r[0] for r in roster}

# attendance containers
first_seen = {}   # roll_no -> first-timestamp (IST)
emotions   = {}   # roll_no -> list[str]

def ist_now():
    # IST = UTC + 5:30
    return (datetime.datetime.utcnow() + datetime.timedelta(hours=5, minutes=30))

def majority(xs):
    from collections import Counter
    if not xs: return ''
    return Counter(xs).most_common(1)[0][0]

# ---- guard: only run in the window ----
if not within_window(ATTN_START[0], ATTN_START[1], ATTN_END[0], ATTN_END[1]):
    print('⛔ Not within 09:30–10:00 IST. Exiting.')
    raise SystemExit

print('✅ Attendance window open. Starting camera… (press Q to stop)')
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera. Try CAM_INDEX=1 or close other apps using the camera.")

while True:
    # if window closes, stop
    if not within_window(ATTN_START[0], ATTN_START[1], ATTN_END[0], ATTN_END[1]):
        print('⏰ Window ended; finalizing CSV…')
        break

    ok, frame = cap.read()
    if not ok:
        print('Camera read failed, stopping.')
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)

    # detect multiple faces
    boxes, _ = mtcnn.detect(pil)
    if boxes is not None:
        for (x1,y1,x2,y2) in boxes:
            x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
            crop = pil.crop((x1,y1,x2,y2)).resize((IMG_SIZE, IMG_SIZE))
            x = preproc(crop).unsqueeze(0)

            with torch.no_grad():
                # FaceID
                face_logits = face_model(x)
                face_prob = F.softmax(face_logits, dim=1)
                face_id = int(face_prob.argmax(1))
                label = face_classes[face_id]  # e.g., '101_AkshayKumar'
                roll_no = label.split('_')[0]

                # Emotion
                emo_logits = emo_model(x)
                emo_id = int(emo_logits.argmax(1))
                emo_label = emo_classes[emo_id]

            # record first seen + emotion stream
            stamp = ist_now().strftime('%Y-%m-%d %H:%M:%S')
            if roll_no in rollnos:
                if roll_no not in first_seen:
                    first_seen[roll_no] = stamp
                emotions.setdefault(roll_no, []).append(emo_label)

            # (Optional visual feedback; not required by assignment)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} | {emo_label}", (x1, max(0,y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

    # comment the next 2 lines if you want NO window at all
    cv2.imshow('Attendance (Q to stop)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('🛑 Stopped by user; finalizing CSV…')
        break

cap.release()
cv2.destroyAllWindows()

# ---- write final CSV ----
rows = []
today = ist_now().strftime('%Y-%m-%d')
for roll_no, name in roster:
    present = roll_no in first_seen
    rows.append({
        'date': today,
        'roll_no': roll_no,
        'name': name,
        'present': 'Yes' if present else 'No',
        'first_seen_time': first_seen.get(roll_no, ''),
        'major_emotion': majority(emotions.get(roll_no, []))
    })

os.makedirs('outputs', exist_ok=True)
outfile = os.path.join('outputs', f"attendance_{today.replace('-','')}.csv")
pd.DataFrame(rows).to_csv(outfile, index=False, encoding='utf-8')
print(f'✅ Saved: {outfile}')

