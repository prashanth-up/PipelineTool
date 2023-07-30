import numpy as np
import pandas as pd
import streamlit as st
import openai
import base64
import io
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Web App Title
st.markdown('''
# **The GPT Extraction Pipeline**

This is the **Data Extraction Pipeline** built for processing our Product store data

App built for Dr. Shang's Lab Group 

---
''')

# Upload CSV data
with st.sidebar.header('1. Upload your data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV or Excel file", type=["csv", "xlsx"])
    output_file = st.sidebar.text_input('Output File Name', value='output', max_chars=None, key=None, type='default')
    output_format = st.sidebar.selectbox('Output Format', ['csv', 'excel', 'both'])

# Function to Load Data
@st.cache
def load_data(file):
    if file.type == 'text/csv':
        data = pd.read_csv(file)
    elif file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        data = pd.read_excel(file)
    else:
        raise ValueError("The file format is not supported. Please upload a CSV or Excel file.")
    return data

def extract_features(df, columns_to_use):
    df = df.copy()  # Create a copy of the dataframe to avoid mutating the cached object.
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

if uploaded_file is not None:
    df = load_data(uploaded_file)
    st.write('---')
    st.header('**Input DataFrame**')
    st.write(df)
    
    # Pandas Profiling Report
    pr = ProfileReport(df, explorative=True)
    st.write('---')
    st.header('**Profiling Report**')
    st_profile_report(pr)

    # GPT-3 Extraction
    st.write('---')
    st.header('**GPT-3 Extraction**')
    api_key = st.text_input('Enter your OpenAI API Key: ')
    openai.api_key = api_key
    all_columns = df.columns.tolist()
    selected_columns = st.multiselect('Select the columns to be used for GPT extraction', all_columns)

    if st.button('Run GPT extraction'):
        # Call the function to extract features using OpenAI GPT
        extracted_df = extract_features(df, selected_columns)

        st.write('---')
        st.header('**GPT Extraction Results**')
        st.dataframe(extracted_df)

        # Save output
        if output_format == 'csv':
            csv = extracted_df.to_csv(index=False)
            st.download_button('Download CSV File', data=csv, file_name=output_file+'.csv', mime='text/csv')
        elif output_format == 'excel':
            towrite = io.BytesIO()
            extracted_df.to_excel(towrite, index=False) 
            towrite.seek(0)  
            st.download_button('Download Excel File', data=towrite, file_name=output_file+'.xlsx', mime='application/vnd.ms-excel')
        else:
            csv = extracted_df.to_csv(index=False)
            st.download_button('Download CSV File', data=csv, file_name=output_file+'.csv', mime='text/csv')
            
            towrite = io.BytesIO()
            extracted_df.to_excel(towrite, index=False) 
            towrite.seek(0)  
            st.download_button('Download Excel File', data=towrite, file_name=output_file+'.xlsx', mime='application/vnd.ms-excel')
else:
    st.info('Awaiting for CSV or Excel file to be uploaded. Please see the sidebar to upload.')
