# Fine Art Painting Genre Classifier
 
**About this project**:

This project explores the use of machine learning techniques in identifying the genre of fine art paintings.

It is divided into six parts:

- **Part 1 Data Collection and Preparation**:
  - Fine art images were downloaded and separated into different folders according to their genre label
  - The images were divided into training, validation, and test sets in preparation for modeling later on
- **Part 2 Transfer Learning**:
  - Pre-trained Convolutional Neural Network (CNN) models such as AlexNet, ResNet-50, and VGG-19 were used
  - Applied transfer learning on the pre-trained models by modifying their final layers and training them to classify fine art paintings based on their genre
- **Part 3 Custom Model**:
  - Implemented a CNN model from scratch to classify fine art paintings based on their genre
- **Part 4 Visualization of Statistics**:
  - Compared the performance of pre-trained models and custom model and visualized their metrics
- **Part 5 Image Segmentation**:
  - Experimented with segmenting objects within a fine art painting using a pre-trained image segmentation model such as DeepLab V3
  - Applied transfer learning on pre-trained image segmentation model to improve performance
- **Part 6 Web App**:
  - Deployed the best-performing classifier model as a web application

## Demo of classifier

The classifier has been deployed as a web application. 

The GIF below demonstrates how to use it:
![](https://github.com/textomatic/fine-art-classifier/blob/main/images/app_demo.gif)

The web app is hosted on Streamlit Cloud. [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fine-art-classifier.streamlit.app) to try out the classifier.

## How to run this project

### Run from the command line

1. Set up the environment using Conda:
```bash
conda env create -n fine-art-classifier -f environment.yml
```

2. Execute the Jupyter notebooks in the folder `notebooks`.

3. Create a separate environment for Streamlit app:
```bash
conda env create -n fine-art-classifier-app -f app/environment.yml
```

4. Launch Streamlit:
```bash
streamlit run app/streamlit_app.py
```