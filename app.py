import streamlit as st
import pandas as pd
from PIL import Image
import subprocess
import os
import base64
import pickle
import joblib


# Molecular descriptor calculator
def desc_calc():
    # Performs the descriptor calculation
    bashCommand = "java -Xms2G -Xmx2G -Djava.awt.headless=true -jar ./PaDEL-Descriptor/PaDEL-Descriptor.jar -removesalt -standardizenitro -fingerprints -descriptortypes ./PaDEL-Descriptor/PubchemFingerprinter.xml -dir ./ -file descriptors_output.csv"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    os.remove("molecule.smi")


# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href


# Model building
def build_model(input_data):
    # Reads in saved regression model
    load_model = model = joblib.load(
        "C:/Users/kusha/OneDrive/Desktop/1mini/model.joblib"
    )

    # Apply model to make predictions
    prediction = load_model.predict(input_data)
    st.header("**Prediction output for Alzehmer's Disease**")
    prediction_output = pd.Series(prediction, name="pIC50")
    molecule_name = pd.Series(load_data[1], name="molecule_name")
    df = pd.concat([molecule_name, prediction_output], axis=1)
    st.write(df)
    st.markdown(filedownload(df), unsafe_allow_html=True)


def build_model_for_cdk2(input_data):
    load_model = joblib.load(
        "C:/Users/kusha/OneDrive/Desktop/1mini/model_cdk2.joblib"
    )
    # Apply model to make predictions
    prediction = load_model.predict(input_data)
    st.header("**Prediction output for Breast Cancer**")
    prediction_output = pd.Series(prediction, name="pIC50")
    molecule_name = pd.Series(load_data[1], name="molecule_name")
    df = pd.concat([molecule_name, prediction_output], axis=1)
    st.write(df.head(5))
    st.markdown(filedownload(df), unsafe_allow_html=True)


def build_model_for_mc(input_data):
    load_model = joblib.load(
        "C:/Users/kusha/OneDrive/Desktop/1mini/model_mc.joblib"
    )
    # Apply model to make predictions
    prediction = load_model.predict(input_data)
    st.header("**Prediction output for Myeloid Cancer**")
    prediction_output = pd.Series(prediction, name="pIC50")
    molecule_name = pd.Series(load_data[1], name="molecule_name")
    df = pd.concat([molecule_name, prediction_output], axis=1)
    st.write(df.head(5))
    st.markdown(filedownload(df), unsafe_allow_html=True)


# Logo image
image = Image.open("logo.png")

st.image(image, use_column_width=True)

# Page title
st.markdown(
    """
# Target Drug Prediction App


An app to predict the bioactivity towards inhibting the enzyme for the target disease.
"""
)


# Sidebar
with st.sidebar.header("1. Upload your CSV data"):
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=["txt"])
    st.sidebar.markdown(
        """
[Example input file](https://raw.githubusercontent.com/Kush-23/bioactivity-prediction-app/main/example_acetylcholinesterase.txt)
""")

if st.sidebar.button("Predict for Myeloid Cancer Disease"):
    load_data = pd.read_table(uploaded_file, sep=" ", header=None)
    load_data.to_csv("molecule.smi", sep="\t", header=False, index=False)

    st.header("**Original input data**")
    st.write(load_data)

    with st.spinner("Calculating descriptors..."):
        desc_calc()

    # Read in calculated descriptors and display the dataframe
    st.header("**Calculated molecular descriptors**")
    desc = pd.read_csv("descriptors_output_mc.csv")
    st.write(desc)
    st.write(desc.shape)

    # Read descriptor list used in previously built model
    st.header("**Subset of descriptors from built models**")
    Xlist = list(pd.read_csv("descriptor_list_mc.csv").columns)
    desc_subset = desc[Xlist]
    st.write(desc_subset)
    st.write(desc_subset.shape)

    # Apply trained model to make prediction on query compounds
    build_model_for_mc(desc_subset)


elif st.sidebar.button("Predict for Alzehmer's Disease"):
    load_data = pd.read_table(uploaded_file, sep=" ", header=None)
    load_data.to_csv("molecule.smi", sep="\t", header=False, index=False)

    st.header("**Original input data**")
    st.write(load_data)

    with st.spinner("Calculating descriptors..."):
        desc_calc()

    # Read in calculated descriptors and display the dataframe
    st.header("**Calculated molecular descriptors**")
    desc = pd.read_csv("descriptors_output.csv")
    st.write(desc)
    st.write(desc.shape)

    # Read descriptor list used in previously built model
    st.header("**Subset of descriptors from built models**")
    Xlist = list(pd.read_csv("descriptor_list.csv").columns)
    desc_subset = desc[Xlist]
    st.write(desc_subset)
    st.write(desc_subset.shape)

    # Apply trained model to make prediction on query compounds
    build_model(desc_subset)


elif st.sidebar.button("Predict for Breast Cancer"):

    load_data = pd.read_table(uploaded_file, sep=" ", header=None)
    load_data.to_csv("molecule.smi", sep="\t", header=False, index=False)

    st.header("**Original input data**")
    st.write(load_data)

    with st.spinner("Calculating descriptors..."):
        desc_calc()

    # Read in calculated descriptors and display the dataframe
    st.header("**Calculated molecular descriptors**")
    desc = pd.read_csv("descriptors_output_cdk2.csv")
    st.write(desc)
    st.write(desc.shape)

    # Read descriptor list used in previously built model
    st.header("**Subset of descriptors from built models**")
    Xlist = list(pd.read_csv("descriptor_list_cdk2.csv").columns)
    desc_subset = desc[Xlist]
    st.write(desc_subset)
    st.write(desc_subset.shape)

    # Apply trained model to make prediction on query compounds
    build_model_for_cdk2(desc_subset)

else:
    st.info("Upload input data in the sidebar to start!")
