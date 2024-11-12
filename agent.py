# agent.py
from langchain_openai import ChatOpenAI 
import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain.agents import AgentExecutor
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import json
from datetime import datetime

from prompt import user_proxy_prompt

load_dotenv()

class CsvAgent:
    def __init__(self, llm, base_path, csv_lst, user_proxy_prompt, csv_prompt):
        self.llm = llm
        self.csv_lst = csv_lst
        self.csv_path_lst = [os.path.join(base_path, x) for x in csv_lst]
        self.pd_lst = [pd.read_csv(x, encoding='utf-8', encoding_errors='replace') 
                      for x in self.csv_path_lst]
        self.user_prompt = user_proxy_prompt
        self.csv_prompt = csv_prompt
        self.cot = ''
        self.parser = StrOutputParser()
        self.max_iterations = 5
        
        # Modified pandas agent configuration
        self.create_pd_agent = create_pandas_dataframe_agent(
            llm=llm,
            df=self.pd_lst[0],
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            prefix=f"""You are an expert data analyst. Process data analysis tasks step by step. Path to the csv file is {self.csv_path_lst[0]}.
                     For each task:
                     1. First describe what you're going to do
                     2. Show your work step by step
                     3. Return results as a properly formatted DataFrame
                     4. If you encounter errors, explain them clearly""",
            suffix="Return entire results in detail in a clear, formatted way.",
            allow_dangerous_code=True
        )
        
        self.initialize_column_info()

    def initialize_column_info(self):
        """Get column information for all dataframes upfront"""
        columns_info = []
        for i, df in zip(self.csv_lst, self.pd_lst):
            sample_data = df.head(1).to_dict('records')[0]
            columns_info.append(f"DataFrame {i}:")
            columns_info.append(f"Columns: {list(df.columns)}")
            columns_info.append(f"Sample data types: {dict((k, type(v).__name__) for k, v in sample_data.items())}\n")
        self.columns_info = "\n".join(columns_info)
        self.cot = f"Initial column information:\n{self.columns_info}\n"

    def preprocess_datetime(self, df):
        """Convert datetime strings to datetime objects"""
        datetime_columns = ['scheduled_departure', 'scheduled_arrival', 'departure_time_local', 'arrival_time_local']
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        return df

    def extract_dataframe_from_result(self, result):
        """Extract DataFrame from agent result if present"""
        if isinstance(result, pd.DataFrame):
            return result
        if isinstance(result, dict) and 'output' in result:
            # Try to find DataFrame in the output string
            output = result['output']
            if isinstance(output, pd.DataFrame):
                return output
            # If output contains DataFrame-like string, try to convert it
            if isinstance(output, str):
                try:
                    # Look for DataFrame representation in the string
                    if 'DataFrame' in output:
                        # Extract the table-like part and convert to DataFrame
                        import io
                        # Find the table-like content between potential explanatory text
                        lines = output.split('\n')
                        table_lines = [line for line in lines if '|' in line or line.strip().startswith('   ')]
                        if table_lines:
                            return pd.read_csv(io.StringIO('\n'.join(table_lines)), sep='|')
                except:
                    pass
        return None

    def user_proxy_agent(self, user_query):
        """Handle user queries with DataFrame return capability"""
        iteration_count = 0
        prompt = PromptTemplate.from_template(self.user_prompt)
        chain = prompt | self.llm | self.parser
        df_result = ''

        while iteration_count < self.max_iterations:
            try:
                res = chain.invoke({
                    'user_query': user_query,
                    'cot': self.cot,
                    'lst_cols': self.columns_info
                })
                
                user_res = json.loads(res)
                agent_response = user_res['agent_response']
                end = user_res['END']

                print(user_res)
                
                if end:
                    return result

                # Process with CSV tool agent
                result = self.csv_tool_agent(user_res)
                
                # Try to extract DataFrame from result
                df_result = self.extract_dataframe_from_result(result)
                if df_result is not None:
                    return df_result
                
                self.cot += f"\nStep {iteration_count + 1}: {agent_response}\nResult: {result}\n"

                user_query = f"Based on this result: {result}, what should we do next to answer: {user_query}"
                
                iteration_count += 1
                
            except Exception as e:
                error_msg = f"Error in iteration {iteration_count}: {str(e)}"
                self.cot += f"\nError: {error_msg}\n"
                return error_msg

        return "Maximum iterations reached. Here's what we found:\n" + self.cot

    def simple_agent(self, user_agent_response):
        try:
            content = user_agent_response["agent_response"]
            
            prompt= f'''
            You are provided with text content and a user query.
            Extract the relevant information of the content using the user query

            user query: {self.user_prompt}

            content: {content}

            '''
            print("End Parse Data", content, self.user_prompt)
            result= llm.invoke(prompt).content
            
            return result
            
        except Exception as e:
            return f"Error in CSV tool agent: {str(e)}"

    def csv_tool_agent(self, user_agent_response):
        """Handle CSV tool operations with DataFrame return"""
        try:
            query = user_agent_response["agent_response"]
            
            result = self.create_pd_agent.invoke({
                "input": f"""
                Analyze this step by step and return results as a DataFrame:
                1. {query}
                2. If calculations are needed, show intermediate steps
                3. Return the final result as a DataFrame object
                """
            })
            
            return result
            
        except Exception as e:
            return f"Error in CSV tool agent: {str(e)}"

    def reset_cot(self):
        """Reset the chain of thought to initial state"""
        self.cot = f"Initial column information:\n{self.columns_info}\n"

# path = "D:/Projects/Final/Portfolio/analyst_agent/processed_files"
# csv_lst = os.listdir(path)

llm = ChatOpenAI(
    model="gpt-4",
    api_key=os.getenv("OPENAI_KEY"),
    temperature=0,
    top_p=0.1
)

