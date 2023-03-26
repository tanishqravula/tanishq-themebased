import streamlit as st
import matplotlib.pyplot as plt 
import os
import numpy as np
import datetime
import itertools
import h5py
import io
from PIL import Image
import tensorflow as tf
from keras.models import load_model
from keras.models import Model

st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def load_cnn1():
    model_ = load_model('C://Users/91770/tanishq malaria streamlit/CNN-Visualization-Using-Streamlit/models/weights1.h5')
    return model_

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def load_cnn2():
    model_ = load_model('C://Users/91770/tanishq malaria streamlit/CNN-Visualization-Using-Streamlit/models/weights3.h5')
    return model_

def preprocessed_image(file):
    image = file.resize((44,44), Image.ANTIALIAS)
    image = np.array(image)
    image = np.expand_dims(image, axis=0) 
    return image

def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1
    st.pyplot(fig)


def main():
    st.title('CNN for Classification Malaria Cells')
    st.sidebar.title('Web Apps using Streamlit')
    st.sidebar.text(""" Project to visualize the CNN layers 
 on malaria-infected image
 
 by Mada Lazuardi Nazilly
 github.com/NazillyMada""")
    menu = {1:"Home",2:"Visualization of Dataset",3:"Perform Prediction"}
    def format_func(option):
        return menu[option]
    choice= st.sidebar.selectbox("Menu",options=list(menu.keys()), format_func=format_func)
    if choice == 1 :
        st.subheader("Dataset Malaria Cells")
        st.markdown("#### Preliminary")
        """ 
        
        This is datasets of segmented cells from the thin blood smear slide images from the Malaria Screener research activity.
        The Dataset is obtained from researchers at the Lister Hill National Center for Biomedical Communications (LHNCBC),
        part of National Library of Medicine (NLM), that developed a mobile application that runs on a standard Android smartphone attached to a conventional light microscope. 
        Giemsa-stained thin blood smear slides from 150 P. falciparum-infected and 50 healthy patients were collected and photographed 
        at Chittagong Medical College Hospital, Bangladesh. The smartphone’s built-in camera acquired images of slides for each microscopic field of view. 
        The images were manually annotated by an expert slide reader at the Mahidol-Oxford Tropical Medicine Research Unit in Bangkok, Thailand. 
        The de-identified images and annotations are archived at NLM (IRB#12972). then applied a level-set based algorithm to detect and segment the red blood cells. 
        The dataset contains a total of 27,558 cell images with equal instances of parasitized and uninfected cells 
        """
        
        st.markdown("#### Previous research")
        """ 
        
        The data appear along with the publications : 
        
        Rajaraman S, Antani SK, Poostchi M, Silamut K, Hossain MA, Maude, RJ, Jaeger S, Thoma GR. (2018) 
        Pre-trained convolutional neural networks as feature extractors toward improved Malaria parasite detection in thin blood smear images.
        
        link : https://peerj.com/articles/4568/ 
        
        Rajaraman S, Jaeger S, Antani SK. (2019) Performance evaluation of deep neural ensembles toward malaria parasite detection in thin-blood smear images
        
        link : https://peerj.com/articles/6977/
        """
        
        st.markdown("#### Links for Malaria Dataset")
        """ 
        More information and download links for this Dataset provided below
        
        https://lhncbc.nlm.nih.gov/publication/pub9932
        
        https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria
        """
    elif choice== 2 :
        st.subheader("Sample Data")
        sample1 = Image.open('C://Users/91770/tanishq malaria streamlit/CNN-Visualization-Using-Streamlit/sample/Capture.png')
        st.image(sample1,caption='Parasitized Cells', use_column_width=True)
        sample2 = Image.open('C://Users/91770/tanishq malaria streamlit/CNN-Visualization-Using-Streamlit/sample/Capture1.png')
        st.image(sample2,caption='Uninfected Cells', use_column_width=True)
        st.markdown("#### Training and Testing Sets")
        """  
        The dataset contains a total of 27,558 cell images with equal instances of parasitized and uninfected cells 
        
        These 2 classes will be separated into training and testing sets. 
        
        Training sets consist of feature_train & label_train.
        Testing sets consist of feature_test & label_test. 
        
        90% for training sets : 10% for testing sets ratio is used
        
        For further explanation, read here : 
        
        https://nazillymada.github.io/Classification-of-Malaria-infected-Cells/
        """
    elif choice == 3 :
        st.subheader("CNN Models")
        st.markdown("#### Complex CNN and Simple CNN")
        """ 
        In this research, we have 2 models of CNN architecture
        
        For further explanation about the architecture and models performance 
        
        read here : https://nazillymada.github.io/Classification-of-Malaria-infected-Cells/
        """
        
        models = st.sidebar.radio(" Select model to perform prediction", ("Complex CNN", "Simple CNN"))
        if models=="Complex CNN":
            model_1 = load_cnn1()
            """ 
            \n ** Complex CNN architecture preview**
            
            Complex CNN is made of 3 blocks of Convolution. 
            
            Consist of 6 conv layers, 4 ReLU-activation layers, 2 Sub-sampling layers, 
            and BatchNormalization + Dropout in the end of every blocks.
            For classification, its using 3 Dense layer with Softmax Activation
            """
            st.subheader('Test on an Image')
            images = st.file_uploader('Upload Image',type=['jpg','png','jpeg'])
            if images is not None:
                images = Image.open(images)
                st.text("Image Uploaded!")
                st.image(images,width=300)
                used_images = preprocessed_image(images)
                predictions = np.argmax(model_1.predict(used_images), axis=-1)
                if predictions == 1:
                    st.error("Cells get parasitized")
                elif predictions == 0:
                    st.success("Cells is healty Uninfected")
                
                st.sidebar.subheader('Visualization in Complex CNN')
                layer_outputs = [layer.output for layer in model_1.layers]
                activation_model = Model(inputs=model_1.input, outputs=layer_outputs)
                activations = activation_model.predict(used_images.reshape((1,44,44,3)))
                layers = st.sidebar.slider('Which layer do you want to see ?', 0, 18, 0, format="no %d ")
                st.subheader('Visualize Layer')
                if layers == 1 :
                    st.write("Layers Conv_1")
                    display_activation(activations, 8, 4, 0)
                elif layers == 2 :
                    st.write("Layers Activation_1 (ReLU)")
                    display_activation(activations, 8, 4, 1)
                elif layers == 3 :
                    st.write("Layers Conv_2")
                    display_activation(activations, 8, 4, 2)
                elif layers == 4 :
                    st.write("Layers Activation_2 (ReLU)")
                    display_activation(activations, 8, 4, 3)
                elif layers == 5 :
                    st.write("Layers Max_pooling2d_1")
                    display_activation(activations, 8, 4, 4)
                elif layers == 6 :
                    st.write("Layers Batch_normalization_1")
                    display_activation(activations, 8, 4, 5)
                elif layers == 7 :
                    st.write("Layers Dropout_1")
                    display_activation(activations, 8, 4, 6)
                elif layers == 8 :
                    st.write("Layers Conv_3")
                    display_activation(activations, 8, 8, 7)
                elif layers == 9 :
                    st.write("Layers Activation_3 (ReLU)")
                    display_activation(activations, 8, 8, 8)
                elif layers == 10 :
                    st.write("Layers Conv_4")
                    display_activation(activations, 8, 8, 9)
                elif layers == 11 :
                    st.write("Layers Max_pooling2d_2")
                    display_activation(activations, 8, 8, 10)
                elif layers == 12 :
                    st.write("Layers Batch_normalization_2")
                    display_activation(activations, 8, 8, 11)
                elif layers == 13 :
                    st.write("Layers Dropout_2")
                    display_activation(activations, 8, 8, 12)
                elif layers == 14 :
                    st.write("Layers Conv_5")
                    display_activation(activations, 16, 8, 13)
                elif layers == 15 :
                    st.write("Layers Activation_4 (ReLU)")
                    display_activation(activations, 16, 8, 14)
                elif layers == 16 :
                    st.write("Layers Conv_6")
                    display_activation(activations, 16, 16, 15)
                elif layers == 17 :
                    st.write("Layers Batch_normalization_3")
                    display_activation(activations, 16, 16, 16)
                elif layers == 18 :
                    st.write("Layers Dropout_3")
                    display_activation(activations, 16, 16, 17)
        
        elif models=="Simple CNN":
            model_2 = load_cnn2()
            
            """ 
            \n **Simple CNN architecture preview**
            
            Simple CNN is made of 1 blocks of Convolution.
            
            Consist only 2 conv layers, 2 Sub-sampling layers 
            and BatchNormalization + Dropout in the end of block.
            for classification, its using 4 Dense layer with Softmax Activation
            """
            st.subheader('Test on an Image')
            images = st.file_uploader('Upload Image',type=['jpg','png','jpeg'])
            if images is not None:
                images = Image.open(images)
                st.text("Image Uploaded!")
                st.image(images,width=300)
                used_images = preprocessed_image(images)
                predictions = np.argmax(model_2.predict(used_images), axis=-1)
                if predictions == 1:
                    st.error("Cells get parasitized")
                elif predictions == 0:
                    st.success("Cells is healty Uninfected")
                
                st.sidebar.subheader('Visualization in Simple CNN')
                layer_outputs = [layer.output for layer in model_2.layers]
                activation_model = Model(inputs=model_2.input, outputs=layer_outputs)
                activations = activation_model.predict(used_images.reshape((1,44,44,3)))
                layers = st.sidebar.slider('Which layer do you want to see ?', 0, 6, 0, format="no %d ")
                st.subheader('Visualize Layer')
                if layers == 1 :
                    st.write("Layers Conv_1")
                    display_activation(activations, 8, 4, 0)
                elif layers == 2 :
                    st.write("Layers Max_pooling2d_1")
                    display_activation(activations, 8, 4, 1)
                elif layers == 3 :
                    st.write("Layers Conv_2")
                    display_activation(activations, 8, 8, 2)
                elif layers == 4 :
                    st.write("Layers Max_pooling2d_2")
                    display_activation(activations, 8, 8, 3)
                elif layers == 5 :
                    st.write("Layers Batch_normalization_1")
                    display_activation(activations, 8, 8, 4)
                elif layers == 6 :
                    st.write("Layers Dropout_1")
                    display_activation(activations, 8, 8, 5)
                
        
if __name__ == "__main__":
    main()
