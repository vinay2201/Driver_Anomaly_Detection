# 🚗 Driver Anomaly Detection using Eye Aspect Ratio (EAR)

This project implements **real-time driver anomaly detection** by monitoring **eye aspect ratio (EAR)** to detect drowsiness using **OpenCV, Dlib, and Scipy**. The system captures live video, detects facial landmarks, and issues an alert if a driver exhibits signs of drowsiness.

---

## 📌 **Project Overview**
Driver fatigue is a major cause of road accidents. This project uses **computer vision and machine learning** to detect drowsiness in real time. The approach involves:
- Detecting a driver's face and extracting eye landmarks using **Dlib**.
- Computing **Eye Aspect Ratio (EAR)** to monitor eye closure.
- Issuing an alert if EAR falls below a threshold, indicating potential drowsiness.

---

## ✅ **Key Features**
✔ **Real-time face and eye detection** using `dlib.get_frontal_face_detector()`.  
✔ **Eye Aspect Ratio (EAR) calculation** to detect eye closure.  
✔ **Drowsiness Alert System**: Displays a warning if the driver is drowsy.  
✔ **Efficient Processing**: Uses OpenCV for optimized video frame analysis.  

---

## 🛠 **Installation & Setup**
1️⃣ **Clone the Repository**
```bash
git clone https://github.com/<your-username>/Driver-Anomaly-Detection.git
cd Driver-Anomaly-Detection
```

2️⃣ **Install Dependencies**
```bash
pip install opencv-python dlib numpy scipy

```
3️⃣ **Download Pre-trained Model The model file shape_predictor_5_face_landmarks.dat is required for facial landmark detection.**

Download the Dlib's Model Repository.
Extract and place it in the project folder.

4️⃣ **Run the Detection Script**
```bash
python driver_detection.py

```
