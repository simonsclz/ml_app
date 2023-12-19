import streamlit as st
import streamlit_image_annotation as sia
from PIL import Image
from time import sleep
import torch
import torchvision.transforms.v2 as transforms


@st.cache_resource
def load_optimizer(_model, lr: float):

    """
    This function loads the optimizer for the model.
    :param _model: The model to load the optimizer for.
    :param lr: The learning rate of the optimizer.
    :return: The torch.optim.Adam optimizer.
    """

    return torch.optim.Adam(_model.parameters(), lr=lr, weight_decay=0.005)


def train(model, image, bboxes, labels):

    """
    This function re-trains the model for one image.
    :param model: The model to re-train.
    :param image: The image to re-train the model on.
    :param bboxes: The bounding boxes in the image.
    :param labels: The labels of the objects in the image.
    :return: None, trains the model.
    """

    labels = [88 for label in labels if label == 0]

    image = transforms.functional.to_tensor(image)
    bboxes = torch.tensor(bboxes)
    labels = torch.tensor(labels)

    # construct the correct bounding box format
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]  # x2 = x1 + width
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]  # y2 = y1 + height

    optimizer = load_optimizer(model, lr=10 ** (-6))

    # set model to training mode
    model.train()

    # workaround to avoid batch size of 1 (batch norm layers would fail)
    losses = model([image] * 2, [{"boxes": bboxes, "labels": labels}] * 2)
    loss = sum(loss for loss in losses.values())

    # backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # reset to evaluation mode
    model.eval()

    print(f"Loss: {loss.item()}")

    # return to the main page
    sleep(2)
    del st.session_state['annotate'], st.session_state['image_to_annotate']
    del st.session_state['result_dict']
    del st.session_state['annotater_created']  # to build a new annotater in the next run
    st.experimental_rerun()


def annotate(model, image_path, image_placeholder):
    """
    This function is used to create the annotater for the Streamlit application.
    :param model: The model used to detect teddy bears in the image.
    :param image_path: The path of the image to annotate.
    :param image_placeholder: The placeholder (Streamlit empty container) for the annotater.
    :return: None. Runs the annotater.
    """

    image_placeholder.empty()
    with image_placeholder.container():
        label_list = ['teddy bear']
        if 'result_dict' not in st.session_state:
            result_dict = {image_path: {'bboxes': [], 'labels': []}}
            st.session_state['result_dict'] = result_dict.copy()

        new_labels = sia.detection(image_path=image_path,
                                   bboxes=st.session_state['result_dict'][image_path]['bboxes'],
                                   labels=st.session_state['result_dict'][image_path]['labels'],
                                   label_list=label_list, line_width=2, key=image_path)
        st.session_state['annotater_created'] = True

        if new_labels is not None:
            st.session_state['result_dict'][image_path]['bboxes'] = [v['bbox'] for v in new_labels]
            st.session_state['result_dict'][image_path]['labels'] = [v['label_id'] for v in new_labels]

            train(model, Image.open(image_path), st.session_state['result_dict'][image_path]['bboxes'],
                  st.session_state['result_dict'][image_path]['labels'])
