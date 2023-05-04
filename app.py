import v5
import v8
import math
import cv2
import streamlit as st
import easyocr
import numpy as np
from deskew import determine_skew
from PIL import Image
reader = easyocr.Reader(['en'])


def preprocess_image(img):
    img = cv2.resize(img, None, fx=1.5, fy=1.2, interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    smooth = cv2.GaussianBlur(img, (1, 1), 0)
    return smooth


def deskew_plate(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = determine_skew(grayscale)
    # print(angle)
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=(0, 0, 0))


def text(img):
    try:
        text = ""
        for ele in reader.readtext(img, allowlist='.-0123456789ABCDEFGHJKLMNPQRSTUVWXYZ'):
            text = text + str(ele[1])
        return text
    except:
        return " "


def detect_objects(img, conf, model_type):
    if model_type == 'v5':
        model = v5.ANPR_V5("models/anpr_v5.pt")
    else:
        model = v8.ANPR_V8("models/anpr_v8.pt")
    plates, image = model.detect(img, conf)
    return image, plates


# Define Streamlit app
def main():
    st.set_page_config(page_title="ANPR using YOLO", page_icon="✨", layout="centered", initial_sidebar_state="expanded")
    st.title(' Automatic Number Plate Recognition 🚘🚙')
    st.write("")
    # selected_type = st.sidebar.selectbox('Please select an activity type 🚀', ["Upload Image", "Live Video Feed"])
    st.sidebar.title('Settings 😎')
    conf = st.sidebar.slider("Confidence Threshold:", 0.0, 1.0, 0.5, step=0.05)

    # Allow user to select model
    model_type = st.sidebar.selectbox("Select Model", ("v5", "v8"))
    top_image = Image.open('static/banner_top.png')
    top_image = st.sidebar.image(top_image,use_column_width='auto')

    # Allow user to upload image
    st.write("")
    uploaded_file = st.file_uploader("Upload an image to process",
                                     type=["jpg", "jpeg", "png"],
                                     help="Supported image formats: JPG, JPEG, PNG")

    if uploaded_file is not None:
        top_image.empty()
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        st.sidebar.text("Uploaded Image:")
        st.sidebar.image(image, caption=" ", use_column_width=True)
        image = deskew_plate(image)
        output_image, plates = detect_objects(image, conf, model_type)
        if len(plates):
            for plate in plates:
                x1, y1, x2, y2, conf = plate
                crop_img = output_image[y1:y2, x1:x2]
                processed_img = preprocess_image(crop_img)
                result = text(processed_img)
                st.sidebar.text('Number plates detected:')
                st.sidebar.image(processed_img, caption=f"Detection Confidence: {conf}", use_column_width=True)
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 255, 255), -1)
                cv2.putText(output_image, f"{result}", (x1 + 7, (y1 + y2) // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 0, 0), 3, cv2.LINE_AA)
        else:
            st.subheader("No License Plates Detected")
        st.image(output_image, caption="Output Image:")


if __name__ == '__main__':
    main()
