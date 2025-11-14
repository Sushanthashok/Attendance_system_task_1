# 🎓 Attendance System with Emotion Detection  

## 📘 Problem Statement

Manual attendance tracking is time-consuming, error-prone, and difficult to scale in large classrooms.
This project automates the process using Face Recognition and Emotion Detection with Deep Learning.

The system:

Detects and identifies students in real time.

Marks Present or Absent automatically.

Detects the emotion of each student.

Logs attendance with timestamp in a .csv or .xlsx file.

Works only during the scheduled class time (e.g. 9:30 AM – 10:00 AM)

## 📂 Dataset

## 1. Student Dataset

A small custom dataset of student face images.

Each student has 5–10 images taken from different angles.

Directory structure:

```
 data/students_raw/
    ├── Student1/
    │   ├── 1.jpg
    │   ├── 2.jpg
    │   └── ...
    ├── Student2/
    └── ...
```

## 2. Emotion Dataset

Public dataset: FER-2013 (Facial Expression Recognition)

Contains 7 emotion classes:

      angry, disgust, fear, happy, neutral, sad, surprise

## 3. Generated Data

During processing, faces are cropped and saved to:

 data/faces_aligned/

 ## ⚙️ Methodology
 
## 🧩 Step 1: Face Detection and Preprocessing

MTCNN (Multi-task Cascaded Convolutional Network) detects and aligns faces.

Cropped face images are resized and normalized for model training.

## 🧠 Step 2: Model Training

(a) Face Identification

MobileNetV2 (PyTorch) is fine-tuned to recognize individual students.

Output: A trained model models/faceid_best.pth.

(b) Emotion Detection

ResNet18 trained on the FER-2013 dataset.

Output: models/emotion_best.pth.

🧾 Step 3: Attendance Logging

When the model detects a known student:

Marks them as Present in an Excel/CSV file.

Records timestamp and emotion.

Ignores unknown faces.

Example CSV output:
```
 Name, Time, Emotion, Status
John, 09:35:12, Happy, Present
Maria, 09:36:10, Neutral, Present
```

## ⏰ Step 4: Time Restriction

Attendance logging is only active between 9:30 AM – 10:00 AM.

Outside this time window, detections are ignored.

## 💻 Model Architecture

| Model             | Base Network | Framework       | Task                 |
| ----------------- | ------------ | --------------- | -------------------- |
| Face Recognition  | MobileNetV2  | PyTorch         | Identify student     |
| Emotion Detection | ResNet18     | PyTorch         | Classify emotion     |
| Face Detection    | MTCNN        | facenet-pytorch | Extract face regions |


## 🧮 Results

| Metric              | Face Recognition | Emotion Detection |
| ------------------- | ---------------- | ----------------- |
| Training Accuracy   | ~87%             | ~82%              |
| Validation Accuracy | ~78%             | ~75%              |
| Inference Speed     | Real-time on CPU | Real-time         |




## Data and Visual Output link
  


👉 DATA : [data](https://drive.google.com/file/d/1TpcKkFufIj0YeJd2Xw5mM79LlLPgYgLL/view?usp=drive_link)

👉 SUCCESSFULLY IMPLEMENTED:[Implementation video](https://drive.google.com/file/d/1ipLvwLTTN1XWxjOOGg9NlrNMH-9uZDny/view?usp=drive_link)




