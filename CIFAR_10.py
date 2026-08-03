#model=load_model('cifar10-object-recognition.keras')
import streamlit as st
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
st.title('CIFAR-10 Dataset')
from tensorflow.keras.models import load_model
#@st.cache_resource
def load_cifar_model():
    model_path=hf_hub_download(repo_id='payal-17/cifar10-model',
filename='cifar10-object-recognition.keras')
    return load_model(model_path)
model=load_cifar_model()
#class name
class_names=['frog', 'truck', 'deer', 'automobile', 'bird', 'horse', 'ship',
       'cat', 'dog', 'airplane']

#image preprocessing
def preprocess_image(image):
    image = Image.open(image)
    image = image.resize((32,32))
    image = image.convert('RGB')      # grayscale
    image = np.array(image)/255.0
    #image = np.expand_dims(image, axis=-1)  # (32,32,1)
    image = np.expand_dims(image, axis=0)   # (1,32,32,1)
    return image

#instruction
upload_image = st.file_uploader('Upload an image', type=['png', 'jpg', 'jpeg'])
if upload_image is not None:
    image = Image.open(upload_image)
    col1,col2=st.columns(2)

    with col1:
        image=image.resize((100,100))
        st.image(image)

    with col2:
        if st.button("Classify"):
            image = preprocess_image(upload_image)

            result = model.predict(image)

            prediction = np.argmax(result)
            prediction_class = class_names[prediction]
            confidence = np.max(result) * 100

            st.success(f"Prediction: {prediction_class}")
            st.write(f"Confidence: {confidence:.2f}%")