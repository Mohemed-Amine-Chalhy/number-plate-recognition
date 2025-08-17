# app.py

import streamlit as st


st.set_page_config(page_title="Smart Gate Control System", layout="wide")

from pathlib import Path
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from PIL import Image



IMAGE_DIR = Path("../images")

preloaded_images = list(IMAGE_DIR.glob("*.jpg"))


# Sidebar selection
selected_image = st.sidebar.selectbox(
    "Choose a preloaded image:",
    preloaded_images,
    format_func=lambda x: x.name
)

st.sidebar.image(str(selected_image), caption=f"Preview: {selected_image.name}", width=250)


st.sidebar.title("About This Project")
st.sidebar.markdown(
    """


    #### Detection Pipeline
    1. **Vehicle Detection** – A YOLO model detects cars in the uploaded image.  
    2. **License Plate Detection** – Another YOLO model finds the plate region.  
    3. **Character Recognition** – A third YOLO model extracts and recognizes each character.  
    4. **OCR Post-Processing** – Characters are sorted left-to-right to reconstruct the full plate number.

    #### 🛠️ How to Use
    - **Choose a preloaded image** from the sidebar list and **drag & drop it** into the uploader.  
    - or Upload one or more **vehicle images** (JPEG, PNG).  
    - The system will **detect cars → plates → characters**.  

    """
)




# -----------------------
# 1. Load Models
# -----------------------
@st.cache_resource
def load_models():
    project_path = Path("..") / Path("models") 
    model1 = YOLO(str(project_path / "yolov10n.pt"))                # Vehicle detection
    model2 = YOLO(str(project_path / "license_plate_detector.pt"))  # Plate detection
    model3 = YOLO(str(project_path / "PlateReaderyolo.pt"))         # Character detection
    return {"model1": model1, "model2": model2, "model3": model3}


models = load_models()
cls_to_letter_map = {'10': 'A', '11': 'B', '12': 'E', '13': 'D', '14': 'H'}


# -----------------------
# 2. Processing Function
# -----------------------
def process_and_visualize_image(img, models, cls_to_letter_map =cls_to_letter_map ):
    
    desired_width = 1024
    aspect_ratio = img.shape[1] / img.shape[0]
    new_height = int(desired_width / aspect_ratio)
    
    img = cv.resize(img, (desired_width, new_height))
    img_with_boxes = img.copy()
  
    
    detected_plates = []  # list of reconstructed plate numbers

    # Stage 0: Detect cars
    result1 = models['model1'](img)[0]
    if not result1.boxes:
        return cv.cvtColor(img_with_boxes, cv.COLOR_BGR2RGB), detected_plates

    for car_box in result1.boxes.xyxy:
        x1_car, y1_car, x2_car, y2_car = map(int, car_box.squeeze())
        cv.rectangle(img_with_boxes, (x1_car, y1_car), (x2_car, y2_car), (0, 0, 255), 2)

        # Stage 1: Detect license plates inside the car
        car_roi = img[y1_car:y2_car, x1_car:x2_car]
        result2 = models['model2'](car_roi)[0]

        if not result2.boxes:
            continue

        for plate_box in result2.boxes.xyxy:
            x1p_rel, y1p_rel, x2p_rel, y2p_rel = map(int, plate_box.squeeze())
            x1p, y1p, x2p, y2p = x1_car + x1p_rel, y1_car + y1p_rel, x1_car + x2p_rel, y1_car + y2p_rel

            cv.rectangle(img_with_boxes, (x1p, y1p), (x2p, y2p), (0, 255, 0), 2)

            plate_img = img[y1p:y2p, x1p:x2p]

         
            # Stage 2: Character recognition
            result3 = models['model3'](plate_img)[0]
            char_boxes = result3.boxes.data.cpu().numpy()

            chars_detected = []
            for char_box in char_boxes:
                x1c, y1c, x2c, y2c, conf, cls = char_box
                x1c_abs, y1c_abs = int(x1p + x1c), int(y1p + y1c)
                x2c_abs, y2c_abs = int(x1p + x2c), int(y1p + y2c)

                cls_name = result3.names.get(int(cls))
                if cls_name in cls_to_letter_map:
                    text = cls_to_letter_map[cls_name]
                
                else :
                    text = cls_name

                
                chars_detected.append((x1c_abs, text))

                cv.rectangle(img_with_boxes, (x1c_abs, y1c_abs), (x2c_abs, y2c_abs), (255, 0, 0), 1)
                cv.putText(img_with_boxes, text, (x1c_abs, y1c_abs - 5),
                        cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

           
            chars_detected.sort(key=lambda x: x[0])
            plate_number = "".join([c[1] for c in chars_detected])
            detected_plates.append(plate_number)
    return cv.cvtColor(img_with_boxes, cv.COLOR_BGR2RGB), detected_plates




# -----------------------
# 3. Streamlit UI
# -----------------------
st.title("🚪 Smart Gate Control System")
st.write("Upload a Moroccan vehicle image to detect the **car → plate → characters**")

uploaded_files = st.file_uploader("Upload one or more images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    for i,uploaded_file in enumerate(uploaded_files):
        st.subheader(f"Car {i + 1} Detection")

        # Read file as OpenCV image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv.imdecode(file_bytes, cv.IMREAD_COLOR)

        # Process image
        processed_img, plates = process_and_visualize_image(img, models, cls_to_letter_map)

        # Show results
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv.cvtColor(img, cv.COLOR_BGR2RGB), caption="Original Image", use_container_width=True)
        with col2:
            st.image(processed_img, caption="Detection Result", use_container_width=True)

        if plates:
            plates = list(set(plates))
            st.success(f"✅ Detected Plate(s): {', '.join(plates)}")
        else:
            st.warning("⚠️ No plate number could be reconstructed.")
