import numpy as np
import pandas as pd
import streamlit as st
import openai
import base64
import io
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


def main():
    st.markdown('''
    # **The GPT Extraction Pipeline**

    This is the **Data Extraction Pipeline** built for processing our Product store data

    **Credit:** App built for Dr. Shang's Lab Group 

    ---
    ''')

    # Upload CSV data
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_files = st.sidebar.file_uploader("Upload your input CSV file", type=["csv", "xlsx"], accept_multiple_files=True)

    # Merge the dataframes from uploaded files
    dfs = []
    for uploaded_file in uploaded_files:
        if ".csv" in uploaded_file.name:
            file = pd.read_csv(uploaded_file)
        elif ".xlsx" in uploaded_file.name:
            file = pd.read_excel(uploaded_file)
        dfs.append(file)

    if dfs:  # if list is not empty
        df = pd.concat(dfs, axis=0, ignore_index=True)
    else:
        df = None
    # df = pd.concat(dfs, axis=0, ignore_index=True)

    # GPT-3
    # with st.sidebar.header('2. GPT-3 Settings'):
    #     openai.api_key = st.sidebar.text_input("OpenAI API Key")
    #     selected_columns = st.sidebar.multiselect("Columns to be included in GPT-3 extraction", options=list(df.columns))
    #     gpt_prompt = st.sidebar.text_input("GPT-3 Prompt", value='TASK: Extract the following features: Strain, Brand, Unit, Flavor, THC, CBD.')
    #     output_file_name = st.sidebar.text_input("Output File Name")
    #     output_format = st.sidebar.selectbox("Output Format", ["csv", "xlsx"])

    with st.sidebar.header('2. GPT-3 Settings'):
        openai.api_key = st.sidebar.text_input("OpenAI API Key")
        selected_columns = st.sidebar.multiselect("Columns to be included in GPT-3 extraction", options=list(df.columns) if df is not None else [])
        gpt_prompt = st.sidebar.text_input("GPT-3 Prompt", value='TASK: Extract the following features: Strain, Brand, Unit, Flavor, THC, CBD.')
        output_file_name = st.sidebar.text_input("Output File Name")
        output_format = st.sidebar.selectbox("Output Format", ["csv", "xlsx"])


    # If no CSV provided, use example data
    if not uploaded_files:
        st.info('Awaiting for CSV file to be uploaded. Currently using example dataset.')
        if st.button('Press to use Example Dataset'):
            example_data = pd.DataFrame(
                np.random.rand(100, 5),
                columns=['a', 'b', 'c', 'd', 'e']
            )
            process_data(example_data, gpt_prompt, selected_columns, output_file_name, output_format)
    else:
        process_data(df, gpt_prompt, selected_columns, output_file_name, output_format)

def extract_features(df, columns_to_use, prompt):
    df = df.copy()
    progress_bar = st.progress(0)
    allowed_columns = ['Strain', 'Brand', 'Unit', 'Flavor', 'THC', 'CBD']  # Define allowed columns

    for i, row in enumerate(df.iterrows()):
        text = ', '.join(str(row[1][col]) for col in columns_to_use)
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt= text + " " + prompt, # Use user-specified prompt
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
                        df.at[row[0], feature] = value.strip()  # Add 'GPT_' prefix
                
            progress_bar.progress((i+1)/len(df))
        except Exception as e:
            st.write(f"Error occurred while processing row {i}: ", e)

    for col in allowed_columns:  # Make sure all the specified columns exist, fill NaN if not present
        if col not in df.columns:
            df[col] = np.nan

    return df

def save_file(df, file_name, output_format):
    if output_format == "csv":
        df.to_csv(file_name+".csv", index=False)
        st.markdown(get_table_download_link_csv(df, file_name), unsafe_allow_html=True)
    else:
        df.to_excel(file_name+".xlsx", index=False)
        st.markdown(get_table_download_link_xlsx(df, file_name), unsafe_allow_html=True)

def get_table_download_link_csv(df, file_name):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}.csv">Download csv file</a>'
    return href


def get_table_download_link_xlsx(df, file_name):
    towrite = io.BytesIO()
    downloaded_file = df.to_excel(towrite, encoding='utf-8', index=False, header=True)
    towrite.seek(0)
    b64 = base64.b64encode(towrite.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}.xlsx">Download excel file</a>'
    return href


def process_data(df, gpt_prompt, selected_columns, output_file_name, output_format):
    st.subheader('Input DataFrame')
    st.write(df)
    if st.button("Run GPT Extraction"):
        st.subheader('GPT-3 Extraction')
        df = extract_features(df, selected_columns, gpt_prompt)
        st.write(df)
        save_file(df, output_file_name, output_format)
    if st.button("Generate Report"):
        st.subheader('Profiling Report')
        pr = ProfileReport(df, explorative=True)
        st_profile_report(pr)

# Remaining functions remain the same ...

if __name__ == '__main__':
    main()
