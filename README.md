# Material Stream Identification System

This project builds a **real-time waste material classifier** that identifies:
`cardboard, glass, metal, paper, plastic, trash`  
from camera images, with an additional **"unknown" (ID = 6)** rejection class.

The system uses:

- **Data augmentation** (flip, rotation, brightness, contrast, blur, zoom)
- **CNN feature extraction** with a pretrained **ResNet-18**
- Two classical ML models:
  - **SVM (RBF kernel)** on CNN features
  - **KNN (k=7, distance-weighted)** on CNN features
- **Rejection mechanisms** to map low-confidence samples to `unknown`
- A **live camera application** for real-time classification

---

## 1. Environment & Installation

```bash
python -m venv venv
venv\Scripts\activate     # on Windows
# or source venv/bin/activate on Linux/Mac

pip install -r requirements.txt
