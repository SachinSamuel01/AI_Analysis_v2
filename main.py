import streamlit as st
import os
from process_csv import clean_data_begin
from agent import CsvAgent, llm
from prompt import user_proxy_prompt
from langchain_openai import ChatOpenAI 

UPLOAD_DIR= r'./uploaded_files/'
PREPROCESS_DIR= r'./processed_files/'

llm = ChatOpenAI(
    model="gpt-4",
    api_key=os.getenv("OPENAI_KEY"),
    temperature=0,
    top_p=0.1
)


def llm_parser(content, query_text):
    prompt= f'''
    You are provided with text content and a user query.
    Extract the relevant information of the content using the user query

    user query: {query_text}

    content: {content}

    '''
    print("End Data", content, query_text)
    res= llm.invoke(prompt).content

    return res

def show_query_page(agent, name):
    st.title(f"Query Interface for {name}")
    
    # Add a back button
    
        
    # Add your query interface elements here
    st.write(f"You can now query the csv files: {name}")
    query_text = st.text_input("Enter your query:")
    if st.button("Run Query"):
        if query_text:
            # Add your query processing logic here
            result = agent.user_proxy_agent(query_text)
            st.write(f"Query: {query_text}")
            st.write(f"Respones: {llm_parser(result, query_text)}")
        else:
            st.warning("Please enter a query")

    if st.button("Back to Main Page"):
        st.session_state.page = "main"
        st.rerun()

def query(name):
    st.session_state.page = "query"
    st.session_state.collection_name = name
    st.rerun()


def save_uploaded_files(uploaded_files, save_dir, name, process_dir):
    """
    Save uploaded files to specified directory with timestamp
    """
    
    
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(process_dir):
        os.makedirs(process_dir)
        
    # Create user-specific directory
    save_dir = os.path.join(save_dir, name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    saved_files = []
    
    for file in uploaded_files:
        if file is not None:
            # Save file
            file_path = os.path.join(save_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            saved_files.append(file.name)
    
    
    process_dir= os.path.join(process_dir, name)
    if not os.path.exists(process_dir):
        os.makedirs(process_dir)

    print(save_dir)
    print(process_dir)
    
    t= clean_data_begin(save_dir, process_dir)
    
    return "uploaded the file", t

def main():

    if 'page' not in st.session_state:
        st.session_state.page = "main"

    if st.session_state.page == "main":
        st.title("Multiple File Upload")
        
        # Add name input
        name = st.text_input("Enter your name:", placeholder="collection name")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose CSV files",
            accept_multiple_files=True,
            type=['csv']
        )
        upload_button =st.button("Upload Files")
        # Upload button
        if  upload_button and uploaded_files and name:
            file_saved_successfully= False
            if len(uploaded_files) > 0:
                # Save directory
                
                
                # Save files
                try:
                    saved_message, clean_file__path = save_uploaded_files(uploaded_files, UPLOAD_DIR, name, PREPROCESS_DIR)
                    
                    # Success message
                    st.success(saved_message)
                    
                    # Show saved files
                    # st.write("Uploaded files:")
                    

                    file_saved_successfully= True
                    
                except Exception as e:
                    st.error(f"Error saving files: {str(e)}")
            else:
                st.warning("Please select at least one file")


            if file_saved_successfully== True:
                st.success(f"After preprocessing the {clean_file__path}")
                
        
        elif upload_button:
            if not name:
                st.warning("Please enter your name")
            if not uploaded_files:
                st.warning("Please select files to upload")

        st.write('---')
        for i in os.listdir(PREPROCESS_DIR):
            if st.button(f"{i}"):
                query(i)

    elif st.session_state.page == "query":
        path= os.path.join(PREPROCESS_DIR,st.session_state.collection_name)
        print(path)
        agent= CsvAgent(llm, path, os.listdir(path), user_proxy_prompt, 'None')
        show_query_page(agent, st.session_state.collection_name)

if __name__ == "__main__":
    main()