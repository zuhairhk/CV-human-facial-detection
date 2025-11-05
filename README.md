# Facial Emotion Detection

## Overview
This project detects **human emotions** from facial expressions using a **hybrid Computer Vision + Machine Learning** approach.  
We use **OpenCV** and **dlib** for facial feature detection, and extracted geometric features are later classified using traditional ML models.

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/zuhairhk/CV-human-facial-detection.git
cd CV-human-facial-detection
```

### 2. Create a virtual environment
```bash
python3 -m venv .venv
```

### 3. Activate the virtual environment
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

### 4. Install dependencies
```bash
pip install opencv-python dlib numpy imutils scikit-learn matplotlib
```

---


---

## Reactivating the Environment

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
