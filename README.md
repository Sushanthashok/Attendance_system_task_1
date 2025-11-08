# 🎓 Attendance System with Emotion Detection  

This project uses **face recognition** and **emotion detection** to automate student attendance marking.  
If a student’s face is detected, the system marks them **present**, identifies their **emotion**,  
and records both details with a **timestamp** in a CSV file.

---

## 🧠 Project Overview
- **Face Detection** → `MTCNN` (from `facenet-pytorch`)
- **Face Recognition (Attendance)** → Custom-trained `MobileNetV2` model
- **Emotion Detection** → Custom-trained `ResNet18` model on the FER2013 dataset
- **Output** → Automatically saves attendance with time and dominant emotion to `outputs/attendance_YYYYMMDD.csv`
- **Time Constraint** → Works only between `09:30 AM – 10:00 AM` IST (can be tested anytime using the `run_attendance_test.py`)
- ## 🔗 Download Large Files
Due to GitHub file size limits, the large folders are uploaded to Google Drive. 

👉 ENVIRONMENT SETUP : [.venv](https://drive.google.com/file/d/136CIAPOB3DehqL2syrtVnLjgYDqhILlM/view?usp=drive_link)

👉 DATA : [data](https://drive.google.com/file/d/1TpcKkFufIj0YeJd2Xw5mM79LlLPgYgLL/view?usp=drive_link)

👉 SUCCESSFULLY IMPLEMENTED:[Implementation video](https://drive.google.com/file/d/1ipLvwLTTN1XWxjOOGg9NlrNMH-9uZDny/view?usp=drive_link)




