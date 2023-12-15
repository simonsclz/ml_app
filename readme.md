# Streamlit application to use an ML model

This project implements a simple Streamlit application
to use an ML model. The model receives an image and displays its predictions.
The user is able to validate the models' predictions or to correct them using
a simple interface.

## How to start the app

There are two ways to run and explore this tiny application.
1. The first method is to glone this repository and run the following command
    in the root directory of the project:
    ```bash
    streamlit run app.py
    ```

2. The second method is to use to visit the project's Streamlit
    cloud website. Therefore, click [this link](https://simple-ml-app.streamlit.app) to get there.

##  References

- The images were taken from the [2017 COCO-Dataset](https://cocodataset.org/#home).
- The used model is a [SSDLITE320_MOBILENET_V3_LARGE](https://pytorch.org/vision/main/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html#torchvision.models.detection.ssdlite320_mobilenet_v3_large)
  from the torchvision package.
- The app uses an [annotations tool](https://github.com/hirune924/Streamlit-Image-Annotation/tree/master)
  that was created by [hirune924](https://github.com/hirune924).
