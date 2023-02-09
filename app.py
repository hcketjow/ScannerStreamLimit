import os
import cv2
import sys
import time
import subprocess
import numpy as np
import streamlit as st
import ssl
# ----------------This part of code has been written by Wojciech Chodasiewicz-----------------------
# This code is for the certificate elements
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# -------------------end--------------------
# ----------------This part of code has been written by ...-----------------------
try:
    import torch
#Read elements about the deep learning
except ModuleNotFoundError as e:
    subprocess.Popen([f"{sys.executable} -m pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu"], shell=True)
    time.sleep(30)
from torchvision.datasets.utils import download_file_from_google_drive
# Download trained models
if not os.path.exists(os.path.join(os.getcwd(), "model_mbv3_iou_mix_2C049.pth")):
    download_file_from_google_drive(file_id=r"1ROtCvke02aFT6wnK-DTAIKP5-8ppXE2a", root=os.getcwd(), filename=r"model_mbv3_iou_mix_2C049.pth")
if not os.path.exists(os.path.join(os.getcwd(), "model_r50_iou_mix_2C020.pth")):
    download_file_from_google_drive(file_id=r"1DEl6qLckFChSDlT_oLUbO2JpN776Qx-g", root=os.getcwd(), filename=r"model_r50_iou_mix_2C020.pth")
# ------------------------------------------------------------
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, deeplabv3_resnet50
from utility_functions import deep_learning_scan, get_image_download_link
# -------------------------end------------------------------------
# ----------------This part of code has been written by Wojciech Chodasiewicz only made some modifiction-----------------------
# Streamlit Components
st.set_page_config(
    page_title="Document Scanner",
    page_icon="", #Here we can set icon of the Firenet.
    layout="centered", #Set whole layout on the center
)
# This will hide the menu on the left
hide_menu_style = """<style>#MainMenu {visibility: hidden;}</style>"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

# This is Select Document Segmentation Backbone Model:
@st.cache(allow_output_mutation=True)
def load_model_DL_MBV3(num_classes=2, device=torch.device("cpu"), img_size=384):
    checkpoint_path = os.path.join(os.getcwd(), "model_mbv3_iou_mix_2C049.pth")
    checkpoints = torch.load(checkpoint_path, map_location=device)
    model = deeplabv3_mobilenet_v3_large(num_classes=num_classes, aux_loss=True).to(device)
    model.load_state_dict(checkpoints, strict=False)
    model.eval()
    with torch.no_grad():
        _ = model(torch.randn((1, 3, img_size, img_size)))
    return model

@st.cache(allow_output_mutation=True)
def load_model_DL_R50(num_classes=2, device=torch.device("cpu"), img_size=384):
    checkpoint_path = os.path.join(os.getcwd(), "model_r50_iou_mix_2C020.pth")
    checkpoints = torch.load(checkpoint_path, map_location=device)
    model = deeplabv3_resnet50(num_classes=num_classes, aux_loss=True).to(device)
    model.load_state_dict(checkpoints, strict=False)
    model.eval()
    with torch.no_grad():
        _ = model(torch.randn((1, 3, img_size, img_size)))
    return model

#I need to change to take better photos
def main(input_file, image_size=16384):
    file_bytes = np.asarray(bytearray(input_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # image = cv2.imdecode(file_bytes, 1)[:, :, ::-1]
    output = None
    model = model_mbv3 if model_selected == "MobilenetV3-Large" else model_r50
    output = deep_learning_scan(og_image=image, trained_model=model, image_size=image_size)
    st.image(output, channels="RGB", use_column_width=True)
    if output is not None:
        st.markdown(
            get_image_download_link(output, f"scanned_{input_file.name}", "Download scanned File"), unsafe_allow_html=True)
    return output

IMAGE_SIZE = 384
model_mbv3 = load_model_DL_MBV3(img_size=IMAGE_SIZE)
model_r50 = load_model_DL_R50(img_size=IMAGE_SIZE)

st.markdown(
    f"""
        <style>
        .stApp{{
             background-image: url("https://media.istockphoto.com/photos/mountain-landscape-picture-id517188688?b=1&k=20&m=517188688&s=612x612&w=0&h=x8h70-SXuizg3dcqN4oVe9idppdt8FUVeBFemfaMU7w=");
             background-attachment: fixed;
             background-size: cover
        }}
        </style>
    """,
    unsafe_allow_html=True
)
st.markdown("<h1 style='text-align: center;'>Document Scanner</h1><br>", unsafe_allow_html=True)
model_selected = st.radio("Select Document Segmentation Model:", ("Mobilenet V3-Large", "ResNet-50"), horizontal=True)
tab1, tab2 = st.tabs(["Capture Document","Upload a Document"] )

st.markdown("""
<style>
    .stTab-selected{
        background-color: white !important;
        color: red;
    }
</style>
""", unsafe_allow_html=True)

with tab2:
    file_upload = st.file_uploader("Upload Document Image :", type=["jpg", "jpeg", "png"])

    if file_upload is not None:
        _ = main(input_file=file_upload, image_size=IMAGE_SIZE)

with tab1:
    run = st.checkbox("Start Camera")
    if run:
        file_upload = st.camera_input("Capture Document", disabled=not run)
        if file_upload is not None:
            _ = main(input_file=file_upload, image_size=IMAGE_SIZE)

