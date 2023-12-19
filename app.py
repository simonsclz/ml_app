from annotater import annotate
import streamlit as st
import numpy as np
from time import sleep
from PIL import Image
import torch
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.v2 as transforms
from torchvision.models.detection import (ssdlite320_mobilenet_v3_large,
                                          SSDLite320_MobileNet_V3_Large_Weights)


@st.cache_resource
def load_random_generator(random_state: int) -> np.random.Generator:
    """
    This function is used to load the random generator from NumPy package.
    :param random_state: The random seed to use.
    :return: NumPy Random Generator.
    """

    return np.random.default_rng(random_state)


@st.cache_resource
def load_model() -> torch.nn.Module:
    """
    Returns the ML model.
    :return: Torchvision SSDLite-model with pre-trained weights.
    """

    return ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)


def detect(model, image: Image, score_threshold: float) -> (Image, int):
    """
    Applies the ML-model to the given image and returns a
    tuple of the image with the marked teddy bears and the number of teddy bears.
    :param model: The ML-model that is going to be applied.
    :param image: The image on which the model is going to be applied.
    :param score_threshold: The threshold for the score of the boxes.
    :return: A tuple of the image with the marked teddy bears and the number of teddy bears.
    """

    model.eval()
    annotations = model([transforms.functional.to_tensor(image)])[0]
    boxes, scores, labels = annotations["boxes"], annotations["scores"], annotations["labels"]

    # just use the teddy bear class
    keep_indices = labels == 88
    boxes, scores, labels = boxes[keep_indices], scores[keep_indices], labels[keep_indices]

    # filter out boxes with a score lower than the threshold
    keep_indices = scores > score_threshold
    boxes, scores, labels = boxes[keep_indices], scores[keep_indices], labels[keep_indices]

    # convert labels to a list of strings
    labels = [f"Teddy bear {scores[index].item()*100:.2f}%" for index, label in
              enumerate(labels) if label.item() == 88]

    # draw the bounding boxes
    image = draw_bounding_boxes(image=transforms.functional.to_image(image),
                                boxes=boxes, labels=labels, width=3, colors="red")
    image = transforms.functional.to_pil_image(image)

    return image, len(boxes)


def correct():
    """
    This function is called when the user clicks on the correct button.
    :return: None.
    """

    st.balloons()
    sleep(2)


def incorrect(image_path: str):
    """
    This function is called when the user clicks on the incorrect button.
    :param image_path: The path of the image that is going to be annotated.
    :return: None.
    """

    st.session_state['annotate'] = True
    st.session_state['image_to_annotate'] = image_path


def app():
    """
    This function is used to create the app for the streamlit dashboard.
    :return: None. Runs the app.
    """

    rg = load_random_generator(42)
    model = load_model()

    if 'annotate' in st.session_state.keys():
        st.error("Oh no! I'm sorry. I'll try better next time. Please help me annotate the image.")

    st.title("Streamlit Teddy Bear Detection 🧸")
    st.markdown("""This is a simple application that deploys a pre-trained ML-model to
    detect teddy bears in images. The model used is a SSDLite320 with a MobilenetV3 backbone from
    the torchvision package. The model was trained on the COCO dataset.
    The user is able to confirm the models' prediction or to correct them.""")
    st.divider()

    image_placeholder = st.empty()

    if 'annotate' in st.session_state.keys():
        image_placeholder.empty()
        st.error("""Please annotate the image.
        Draw the bounding boxes and confirm by clicking the 'Complete' button.
        Thank you for your help!""")
        annotate(model, st.session_state['image_to_annotate'])

    else:
        image_number = rg.integers(1, 5, None, endpoint=True)
        img = Image.open(f"images/{image_number}.jpg")
        (img, num) = detect(model, img, 0.5)

        image_placeholder.empty()
        with image_placeholder.container():
            st.image(img)

        st.divider()

        if num == 1:
            st.markdown("### I've detected **1** teddy bear. Look at the box. Am I correct?")
        else:
            st.markdown(f"### I've detected **{num}** teddy bears. Look at the boxes. Am I correct?")

        c1, c2 = st.columns(2)
        c1.button("Correct ✅", on_click=correct)
        c2.button("Incorrect ❌", on_click=incorrect,
                  args=(f"images/{image_number}.jpg",))


if __name__ == "__main__":
    app()
