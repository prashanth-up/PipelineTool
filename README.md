# GPT Extraction Pipeline README

## Overview

The GPT Extraction Pipeline is a Data Extraction Pipeline designed for processing product store data. Developed for Dr. Shang's Lab Group, this tool integrates Python libraries such as NumPy, Pandas, Streamlit, OpenAI, and Pandas Profiling to provide a comprehensive data processing solution. It allows users to upload data, select relevant columns for GPT-3 data extraction, and generate reports for further analysis.

## Features

- Data upload from CSV or XLSX files with support for multiple files.
- Dataframe merging for uploaded files.
- GPT-3 integration for data extraction based on selected columns.
- Option to use an example dataset if no data is uploaded.
- Data extraction feature customization through a user-defined GPT-3 prompt.
- Data export in CSV or XLSX format with download link generation.
- Data profiling report generation for exploratory data analysis.

## Requirements

To run the GPT Extraction Pipeline, ensure you have the following libraries installed:

- numpy
- pandas
- streamlit
- openai
- ydata_profiling
- streamlit_pandas_profiling

## Installation

First, make sure you have Python installed on your system. Then, install the required libraries using pip:

```
pip install numpy pandas streamlit openai streamlit-pandas-profiling
```

## Usage
To start the application, navigate to the directory containing the script and run:

```
Copy code
streamlit run your_script_name.py
Replace your_script_name.py with the actual name of your Python script.
```

#### Steps:
- Upload Data: Begin by uploading your CSV or XLSX data files through the sidebar.
- GPT-3 Settings: Configure the GPT-3 settings including API key, columns for extraction, prompt, output file name, and format.
- Data Processing: Use the provided buttons to run GPT extraction or generate a profiling report based on the uploaded data.

## Contribution
Contributions to the GPT Extraction Pipeline are welcome. Please fork the repository, make your changes, and submit a pull request for review.

## License
This project is currently licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
Credit: App built for Dr. Shang's Lab Group
Special thanks to the open-source libraries used in this project.

For any issues or further inquiries, please contact the project maintainers.
