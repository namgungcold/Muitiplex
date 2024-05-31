
import streamlit as st
import cv2
from PIL import ImageEnhance



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CyclicLR
import torchvision.utils as vutils


from utils.loss import ContentLoss, AdversialLoss
from utils.transforms import get_default_transforms, get_no_aug_transform
from utils.datasets import get_dataloader
from utils.transforms import get_pair_transforms
from torch.utils.tensorboard import SummaryWriter
from models.discriminator import Discriminator
from models.generator import Generator

from datetime import datetime
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import pickle
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# netD=Discriminator().to(device)
# netG=Generator().to(device)

# netG.load_state_dict(torch.load("./checkpoints/trained_netG_original.pth"))
# netD.load_state_dict(torch.load("./checkpoints/trained_netD_original.pth"))



# Load the pre-trained Haar Cascade classifiers
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
except Exception:
    st.write("Error loading cascade classifiers")

def test_image(image):
    # name = file.split(".")[0]
    netG = Generator().to(device)
    netG.load_state_dict(torch.load("./checkpoints/trained_netG_original.pth"))
    trf = get_no_aug_transform()
    image = trf(image)
    image = image.unsqueeze(0).to(device)
    netG.eval()
    with torch.no_grad():
        pred_image = netG(image)
    # netG.eval()

    img = np.transpose(vutils.make_grid(pred_image.detach().cpu(), normalize=True).cpu(), (1, 2, 0)).numpy()
    # cv2.imwrite(f"./result/{name}.png", cv2.cvtColor(img * 255, cv2.COLOR_RGB2BGR))
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.medianBlur(gray, 5)
    netG = []
    # edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    return img
#
# def test_image2(image):
#     # image_numpy = np.array(image)
#     image=test_image(image)
#     image_numpy = np.array(image)
#     trf=get_no_aug_transform()
#     img=np.transpose(vutils.make_grid(pred_image.detach().cpu(), normalize=True).cpu(), (1, 2, 0)).numpy()
#     return img


def detect_faces(image):
    img = np.array(image.convert("RGB"))
    # Detect faces
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        roi = img[y:y+h, x:x+w]

        # Detect eyes in the face(s) detected
        eyes = eye_cascade.detectMultiScale(roi)

        # Detect smiles in the face(s) detected
        smile = smile_cascade.detectMultiScale(roi, minNeighbors = 25)

        # Draw rectangle around eyes
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)

        # Draw rectangle around smile
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi, (sx, sy), (sx+sw, sy+sh), (0,0,255), 2)

    # Returning the image with bounding boxes drawn on it and the counts
    return img, faces

def cartoonize_image(image):
    img = np.array(image.convert("RGB"))
    img = cv2.cvtColor(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def cannize_image(image):
    img = np.array(image.convert("RGB"))
    img = cv2.cvtColor(img, 1)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    canny = cv2.Canny(img, 100, 150)
    return canny

def main():

    # Page layout
    st.set_page_config(page_title="Image convert",
    layout="wide",
    initial_sidebar_state="collapsed")

    # Title and description
    st.title("Image convert 🖼️")
    st.sidebar.title("Main")
    # st.sidebar.title("Image convert 🖼️")
    t=st.sidebar.selectbox("Select",["Example",'Converter'])

    #image_file = st.sidebar.file_uploader("Upload image", type=["jpg","png","jpeg"])


    # task = ["Image Enhancement", "Image Detection"]
    # choice = st.sidebar.selectbox("Choose task", task)
    width = max(50, 0.01)
    side = max((100 - 50) / 2, 0.01)
    _, container3, _,_, container4, _ = st.columns([width, width,width, width, width, width])
    container1, container2 = st.columns([width, width])

    if t=='Example':

        # images = ['./picture/karina.jfif', './result/karina.png']
        # st.image(images, use_column_width=False, width=550 )

        # st.subheader("Image")
        # st.write("변환 전")
        container3.subheader("  변환 전")
        container4.subheader("  변환 후")
        
        
        container1.image(Image.open('./picture/karina.jfif'))
        # st.write('변환 후')
        container2.image(Image.open('./result/karina.png'))

        DEFAULT_WIDTH = 80
        # VIDEO_DATA =['./final_ori.mp4','./test_video.mp4']

        # width = st.sidebar.slider(
        #     label="Width", min_value=0, max_value=100, value=DEFAULT_WIDTH, format="%d%%"
        # )


        container1.video(data='./final_ori.mp4')
        container2.video(data='./test_video.mp4')



        # st.subheader("Video")
        # st.write('변환 전')
        # video_file = open('./final_ori.mp4', 'rb')
        # video_bytes = video_file.read()
        # st.video(video_bytes,start_time=1)
        #
        # st.write("변환 후")
        # convert_video_file = open('./test_video.mp4', 'rb').read()
        # st.video(convert_video_file,start_time=1)

    # Open and preview original image
    else:
        image_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg",'jfif'])

        if image_file is not None:
            image = Image.open(image_file)
            st.subheader("Original")
            st.image(image)

            # Image enhancement
            # if choice == "Image Enhancement":

            types = ["Gray-Scale", "Contrast", "Brightness", "Color Balance", "Blur", "Cartoonize",'Cartoon_Gan',"Cartoon_gan_black"]
            enhance_type = st.sidebar.radio("Convert Type",types)

            st.subheader("Result")
            # Gray-scale
            if enhance_type == "Gray-Scale":
                new_img = np.array(image.convert("RGB"))
                img = cv2.cvtColor(new_img, 1)
                gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                st.image(gray)

            # Contrast
            elif enhance_type == "Contrast":
                contrast_rate = st.sidebar.slider("Contrast Rate", 0.5, 3.0, step=0.1)
                enhancer = ImageEnhance.Contrast(image)
                img_output = enhancer.enhance(contrast_rate)
                st.image(img_output)

            # Brightness
            elif enhance_type == "Brightness":
                brightness_rate = st.sidebar.slider("Brightness Rate", 0.5, 3.0, step=0.1)
                enhancer = ImageEnhance.Brightness(image)
                img_output = enhancer.enhance(brightness_rate)
                st.image(img_output)

            # Color balance
            elif enhance_type == "Color Balance":
                color_balance = st.sidebar.slider("Color Balance", 0.5, 3.0, step=0.1)
                enhancer = ImageEnhance.Color(image)
                img_output = enhancer.enhance(color_balance)
                st.image(img_output)

            # Blur
            elif enhance_type == "Blur":
                new_img = np.array(image.convert("RGB"))
                blur_rate = st.sidebar.slider("Blur Rate", 0.5, 3.0, step=0.1)
                img = cv2.cvtColor(new_img,1)
                blur_img = cv2.GaussianBlur(img, (11,11), blur_rate)
                st.image(blur_img)

            # Cartoonize
            elif enhance_type == "Cartoonize":
                result_img = cartoonize_image(image)
                st.image(result_img)

            # Cartoon_gan
            elif enhance_type == "Cartoon_Gan":
                image = image.convert('RGB')
                result_cartoon=test_image(image)
                st.image(result_cartoon)

            # Cartoon_gan_black
            elif enhance_type == "Cartoon_gan_black":
                image = image.convert('RGB')
                result_cartoon = test_image(image)
                img = cv2.cvtColor(result_cartoon, 1)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                st.image(gray)



        # else:
        #     # Image detection
        #     detector_list = ["Face Detector", "Canny Edge Detector"]
        #     detector_choice = st.sidebar.radio("Select Detector", detector_list)
        #     if st.sidebar.button("Process"):
        #         st.subheader("Result")
        #
        #         # Face detector
        #         if detector_choice == "Face Detector":
        #             result_img, result_faces = detect_faces(image)
        #             st.image(result_img)
        #             if len(result_faces)> 1:
        #                 st.success(f"Found {len(result_faces)} faces")
        #             else:
        #                 st.success(f"Found {len(result_faces)} face")
        #
        #         # Canny Edge Detector
        #         elif detector_choice == "Canny Edge Detector":
        #             result_img = cannize_image(image)
        #             st.image(result_img)

    # else:
    #     st.info("Please upload an image to get started.")

if __name__ == "__main__":
    main()
