import os
import pandas as pd
import numpy as np
import openai
import streamlit as st
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

st.markdown('''
# **The GPT Extraction Pipeline**

This is the **Data Extraction Pipeline** built for processing our Product store data

App built for Dr. Shang's Lab Group 

---
''')

uploaded_file = st.sidebar.file_uploader("Upload your input CSV or Excel file", type=["csv", "xlsx"])

api_key = st.text_input('Insert your OpenAI API key here')

if api_key:
    openai.api_key = api_key

def load_file():
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        return df
    except Exception as e:
        st.write('Error occurred while loading data: ', e)
        
def generate_profile(df):
    try:
        pr = ProfileReport(df, explorative=True)
        return pr
    except Exception as e:
        st.write('Error occurred while generating profile: ', e)

def extract_features(df, columns_to_use):
    progress_bar = st.progress(0)
    allowed_columns = ['Strain', 'Brand', 'Unit', 'Flavor', 'THC', 'CBD']  # Define allowed columns
    for i, row in enumerate(df.iterrows()):
        text = ', '.join(str(row[1][col]) for col in columns_to_use)
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt= text + " TASK: Extract the following features: Strain; Brand; Unit; Flavor; THC; CBD. Report them in a structured way like - 'Strain=<value>; Brand=<value>; Unit=<value>; Flavor=<values>; THC=<value>; CBD=<value>' and the '<value>' can be 'none' if there was no information in the text",
                max_tokens=2048,
                n=1,
                stop=None,
                temperature=0.5,
            )
            
            extracted_data = response["choices"][0]["text"].strip().split(';')
            for data in extracted_data:
                if '=' in data:
                    feature, value = data.split('=')
                    feature = feature.strip()
                    if feature in allowed_columns:  # Only allow defined columns
                        df.at[row[0], feature] = value.strip()
                
            progress_bar.progress((i+1)/len(df))
        except Exception as e:
            st.write(f"Error occurred while processing row {i}: ", e)
    return df



def save_file(df, filename, fileformat):
    try:
        if fileformat == 'CSV':
            df.to_csv(f"{filename}.csv", sep=',', encoding='utf-8')

            # df.to_csv(f"{filename}.csv", sep='\t', encoding='utf-8')
        elif fileformat == 'Excel':
            df.to_excel(f"{filename}.xlsx", index=False)
        st.success(f'File saved as {fileformat} successfully.')
    except Exception as e:
        st.write(f'Error occurred while saving data as {fileformat}: ', e)

if uploaded_file is not None:
    df = load_file()
    if df is not None:
        st.header('**Input DataFrame**')
        st.write(df)
        st.write('---')
        pr = generate_profile(df)
        if pr is not None:
            st.header('**Profiling Report**')
            st_profile_report(pr)
        
        if api_key:
            selected_columns = st.multiselect('Select columns to use for GPT extraction', df.columns)
            filename = st.text_input('Input the output file name')
            fileformat = st.selectbox('Select the output file format', ['CSV', 'Excel'])
            
            if st.button('Run GPT extraction'):
                df = extract_features(df.head(5), selected_columns)
                st.write(df)
                if filename:
                    save_file(df, filename, fileformat)
                else:
                    st.error('Please provide a file name for the output file.')
        else:
            st.info('Please provide an OpenAI API key to extract features.')
else:
    st.info('Awaiting for CSV or Excel file to be uploaded.')
