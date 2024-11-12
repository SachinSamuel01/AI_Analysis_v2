# CSV Data Analysis Assistant
A Streamlit-based web application that allows users to upload CSV files, process them, and perform natural language queries on the data using OpenAI's GPT-4 model.
Features

- File Upload: Upload multiple CSV files simultaneously
- Data Processing: Automatic preprocessing of uploaded CSV files
- Collection Management: Create and manage different collections of CSV files
- Natural Language Queries: Ask questions about your data in plain English
- Interactive Interface: Clean and user-friendly web interface
- Smart Response Parsing: LLM-powered interpretation of query results

## Prerequisites
Before running the application, make sure you have:

- Python 3.8 or higher
- OpenAI API key
- Required Python packages (see Installation section)

## Installation

Clone the repository:
```bash
bashCopygit clone <repository-url>
cd <repository-name>
```

Create a virtual environment (recommended):
```bash
python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate
```

Install required packages:
```bash
pip install -r requirements.txt
```

Create a .env file in the root directory and add your OpenAI API key:
```bash
OPENAI_KEY=your-api-key-here
```

## Project Structure
```bash
project_root
├── agent.py              # CSV analysis agent implementation
├── process_csv.py        # CSV preprocessing functions
├── prompt.py            # LLM prompt templates
├── main.py               # Main Streamlit application
├── uploaded_files/      # Directory for raw uploaded files
├── processed_files/     # Directory for preprocessed files
└── requirements.txt     # Project dependencies
```

## Usage

Start the application:
```bash
streamlit run main.py
```

Access the application in your web browser (typically at http://localhost:8501)

Using the application:

- Enter a collection name
- Upload one or more CSV files
- Wait for preprocessing to complete
- Click on your collection name to access the query interface
- Enter natural language queries about your data
View the parsed and formatted results
