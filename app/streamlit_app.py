import streamlit as st
import io
import os
import warnings
import torch
from torchvision import transforms
from PIL import Image
warnings.filterwarnings('ignore')

MODEL_PATH = os.path.dirname(os.path.realpath(__file__)) + '/fine_art_classifier_model.pt'
LABELS_PATH = os.path.dirname(os.path.realpath(__file__)) + '/model_classes.txt'


@st.cache()
def load_model(model_path, categories):
    model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg19', pretrained=True)
    model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, len(categories))
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


@st.cache()
def load_labels(labels_path):
    with open(labels_path, 'r') as f:
        categories = [s.strip() for s in f.readlines()]
        return categories


def load_image():
    uploaded_file = st.file_uploader(label='Upload an image of a fine art painting to test:')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def predict(model, categories, image):
    preprocess_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess_transforms(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    all_prob, all_catid = torch.topk(probabilities, len(categories))

    for i in range(all_prob.size(0)):
        st.write(categories[all_catid[i]], all_prob[i].item())


if __name__ == '__main__':
    st.set_page_config(page_title='Fine Art Genre Classifier')
    st.title('Fine Art Genre Classifier')
    st.subheader('This web app classifies the genre of a fine art painting')
    categories = load_labels(LABELS_PATH)
    model = load_model(MODEL_PATH, categories)
    image = load_image()
    result = st.button('Classify image')
    if result:
        st.write('Calculating results...')
        predict(model, categories, image)