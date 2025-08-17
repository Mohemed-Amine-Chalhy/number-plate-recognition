

# 🚪 Smart Gate Control System

A Streamlit-based application for **automatic vehicle and license plate detection**, specifically tailored for Moroccan plates. The system detects cars, identifies license plates, and recognizes individual characters to reconstruct plate numbers.

---

[![Screenshot-2025-08-17-172654.png](https://i.postimg.cc/c421V4w4/Screenshot-2025-08-17-172654.png)](https://postimg.cc/WhmPZV5Q)
---

## 🗂️ Repository Structure

```

project-root/
│
├─ app/
│   └─ app.py                  # Main Streamlit app
│
├─ jupyter\_notebooks/
│   └─ ...                     # Notebooks for experiments, testing, or preprocessing
│
├─ images/                     # Preloaded test images
│   └─ \*.jpg
│
├─ models/                     # YOLO models
│   ├─ yolov10n.pt             # Vehicle detection
│   ├─ license\_plate\_detector.pt  # License plate detection
│   └─ PlateReaderyolo.pt      # Character recognition
│
└─ README.md

````

---

## ⚙️ Features

- **Vehicle Detection:** Detect cars in uploaded images using YOLO.
- **License Plate Detection:** Detect plates inside cars.
- **Character Recognition:** Recognize individual letters and numbers on plates.
- **OCR Post-processing:** Sort characters left-to-right to reconstruct the full plate number.
- **Preloaded Images:** Drag & drop preloaded images from the sidebar for quick testing.
- **Supports Uploads:** Upload one or more images (JPEG, PNG) via drag & drop.

---

## 🛠️ How to Use

1. Launch the app:

```bash
cd app
streamlit run app.py
````

2. **Sidebar Options:**

   * Select a preloaded image from the list and **drag & drop it** into the uploader.
   * Or upload your own vehicle images.

3. The system will automatically:

   * Detect the vehicle
   * Detect the license plate
   * Recognize characters and reconstruct the plate number

4. View results side by side:

   * Original image
   * Detection result with bounding boxes and recognized characters

---

## 📦 Requirements

* Python 3.10+
* Streamlit
* OpenCV (`cv2`)
* PIL (`Pillow`)
* NumPy
* [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)

Install dependencies:

```bash
pip install streamlit opencv-python pillow numpy ultralytics
```

---

## 🧪 Notes

* The system is optimized for **Moroccan license plates**, but can be adapted for other plate formats.
* Ensure the models are placed in the `models/` folder as described above.
* For better performance, a GPU is recommended when running YOLO models.

---

