import streamlit as st
import pandas as pd
import pickle
import time
from PIL import Image

st.set_page_config(page_title="Halaman Modelling", layout="wide")
st.write("""
# Welcome to my portofolio Data Analyst


""")
add_selectitem = st.sidebar.selectbox("Want to open about?", ("Iris species!", "Heart Disease!"))

def iris():
    st.write("""
    Develop by [Bramantio](https://www.linkedin.com/in/brahmantio-w/)
    
    Data obtained from the [iris dataset](https://www.kaggle.com/uciml/iris) by UCIML. 
    """)
    st.sidebar.header('User Input Features:')
    
    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            st.sidebar.header('Input Manual')
            SepalLengthCm = st.sidebar.slider('Sepal Length (cm)', 4.3,10.0,6.5)
            SepalWidthCm = st.sidebar.slider('Sepal Width (cm)', 2.0,5.0,3.3)
            PetalLengthCm = st.sidebar.slider('Petal Length (cm)', 1.0,9.0,4.0)
            PetalWidthCm = st.sidebar.slider('Petal Width (cm)', 0.1,5.0,1.4)
            data = {'SepalLengthCm': SepalLengthCm,
                    'SepalWidthCm': SepalWidthCm,
                    'PetalLengthCm': PetalLengthCm,
                    'PetalWidthCm': PetalWidthCm}
            features = pd.DataFrame(data, index=[0])
            return features
    
    input_df = user_input_features()
    img = Image.open("iris.JPG")
    st.image(img, width=500)
    if st.sidebar.button('Predict!'):
        with open("model_iris.pkl", 'rb') as file:
            st.write(input_df)
            model = pickle.load(file)
            prediction = model.predict(input_df)
            result = ['Iris-setosa' if prediction == 0 else ('Iris-versicolor' if prediction == 1 else 'Iris-virginica')]
        st.subheader('Prediction: ')
        output = str(result[0])
        with st.spinner('Wait for it...'):
            time.sleep(4)
            st.success(f"Prediction of this app is {output}")
            
def heart():
    st.write("""
    This app predicts the **Heart Disease**
    
    Data obtained from the [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by UCIML. 
    """)
    st.sidebar.header('User Input Features:')
    # Collects user input features into dataframe
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features1():
            st.sidebar.header('Manual Input')
            age = st.sidebar.slider("Usia", 29, 77, 30)
            sex = st.sidebar.selectbox("Jenis Kelamin", ('Perempuan', 'Pria'))
            if sex == "Perempuan":
                sex = 0
            else:
                sex = 1 
            cp = st.sidebar.slider('Chest pain type', 1,4,2)
            if cp == 1.0:
                wcp = "Nyeri dada tipe angina"
            elif cp == 2.0:
                wcp = "Nyeri dada tipe nyeri tidak stabil"
            elif cp == 3.0:
                wcp = "Nyeri dada tipe nyeri tidak stabil yang parah"
            else:
                wcp = "Nyeri dada yang tidak terkait dengan masalah jantung"
            st.sidebar.write("Jenis nyeri dada yang dirasakan oleh pasien", wcp)
            trestbps = st.sidebar.slider ("Tingkat BPS Pasien", 94, 200, 94)
            chol = st.sidebar.slider("Tingkat Kolesterol Pasien", 126, 564, 126)
            fbs = st.sidebar.slider("Tingkat Kadar gula pasien", 0, 1, 0)
            restecg = st.sidebar.slider("EKG Pasien Saat Istirahat", 0, 2, 0)
            thalach = st.sidebar.slider("Maximum heart rate achieved", 71, 202, 80)
            exang = st.sidebar.slider("Exercise induced angina", 0, 1, 1)
            oldpeak = st.sidebar.slider("Seberapa banyak ST segmen menurun atau depresi", 0.0, 6.2, 1.0)
            slope = st.sidebar.slider("Kemiringan segmen ST pada elektrokardiogram (EKG)", 0, 2, 1)
            ca = st.sidebar.slider("Number of major vessels", 0, 3, 1)
            thal = st.sidebar.slider("Hasil tes thalium", 1, 3, 1)
            data = {'age':age,
                    'sex': sex,
                    'cp': cp,
                    'trestbps':trestbps,
                    'chol':chol,
                    'fbs':fbs,
                    'restecg':restecg,
                    'thalach': thalach,
                    'exang': exang,
                    'oldpeak': oldpeak,
                    'slope': slope,
                    'ca':ca,
                    'thal':thal,
                   }
            features = pd.DataFrame(data, index=[0])
            return features
    
    input_df = user_input_features1()
    img = Image.open("heart-disease.jpg")
    st.image(img, width=500)
    if st.sidebar.button('Predict!'):
        with open("modelheart.pkl", 'rb') as file:
            st.write(input_df)
            model = pickle.load(file)
            prediction = model.predict(input_df)        
            result = ['No Heart Disease' if prediction == 0 else 'Yes Heart Disease']
        st.subheader('Prediction: ')
        output = str(result[0])
        with st.spinner('Wait for it...'):
            time.sleep(4)
            st.success(f"Prediction of this app is {output}")
         
if add_selectitem == "Iris species!":
    iris()
elif add_selectitem == "Heart Disease!":
    heart()
