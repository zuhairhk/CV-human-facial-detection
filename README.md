# 🧠 Facial Emotion Detection (Hybrid CV + ML)

## Overview
This project detects **human emotions** from facial expressions using a **hybrid Computer Vision + Machine Learning** approach.  
We use **OpenCV** and **dlib** for facial feature detection, and extracted geometric features are later classified using traditional ML models.

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/cv843-emotion-detection.git
cd cv843-emotion-detection
```

### 2️⃣ Create a virtual environment
```bash
python3 -m venv .venv
```

### 3️⃣ Activate the virtual environment
- **Linux / macOS**
  ```bash
  source .venv/bin/activate
  ```
- **Windows (PowerShell)**
  ```bash
  .venv\Scripts\activate
  ```

You should now see `(.venv)` appear before your terminal prompt.

---

### 4️⃣ Install dependencies
```bash
pip install opencv-python dlib numpy imutils scikit-learn matplotlib
```

> 🧩 If `dlib` fails to install, make sure you have build tools:
> ```bash
> sudo apt install cmake g++ python3-dev
> ```

---

### 5️⃣ Download required model file
Download the pre-trained facial landmark model:

📦 **[shape_predictor_68_face_landmarks.dat](https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2)**  
Then extract it:
```bash
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```
and place it in the **root** of this project directory.

---

## ▶️ Running the Code

### Detect facial landmarks in an image:
```bash
python detect_landmarks.py --image path/to/image.jpg
```

### (Optional) Real-time webcam test:
```bash
python detect_landmarks_live.py
```

---

## 🔁 Reactivating the Environment (Every Time You Work)

Each time you reopen the project:
```bash
cd ~/Documents/Code/CV-human-facial-detection
source .venv/bin/activate
```

When you’re done:
```bash
deactivate
```

---
