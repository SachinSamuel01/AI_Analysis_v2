from langchain_openai import ChatOpenAI 
import os
from dotenv import load_dotenv
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.prompts import PromptTemplate
import json
from typing import List, Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import BaseTool
import re



from prompt import planning_prompt, synthesis_prompt, prefix_prompt

load_dotenv()

class CustomTool(BaseTool):
    name:str = "custom_tool"
    description:str = "call to find whether the pilot is new or old. You pass the name and it return True if pilot is new else False."
    
    def old_or_new_pilot(self, name: str) -> bool:
        if name[-1] == 'a':
            return False
        else:
            return True
    def _run(self, name: str) -> bool:
        return self.old_or_new_pilot(name)
    





llm_gemini= ChatGoogleGenerativeAI(
    model= "gemini-1.5-flash",
    temperature= 0,
    top_p= 0.1,
    api_key= os.getenv('GEMINI_API_KEY')
)




def clean_and_extract_json(  text):
    # First pattern to match and remove ```json or ```JSON and their closing ```
        code_block_pattern = r'```(?:json|JSON)\n(.*?)```'

        # Pattern to match "user_res: " prefix if it exists
        prefix_pattern = r'^user_res:\s*'

        def process_text(input_text):
            # Remove prefix if it exists
            text_without_prefix = re.sub(prefix_pattern, '', input_text.strip())

            # Check if we have a code block
            code_block_match = re.search(code_block_pattern, text_without_prefix, re.DOTALL)
            if code_block_match:
                # If we found a code block, return its contents
                return code_block_match.group(1).strip()
            else:
                # If no code block markers, return the cleaned text
                return text_without_prefix.strip()
        return process_text(text)


def add_thought(step: str, result: Any, thoughts:list):
        """Record a step in the chain of thought"""
        thoughts.append({
            "step": step,
            "result": str(result),
            "timestamp": pd.Timestamp.now()
        })

def plan_analysis(columns, query: str,thoughts, des) -> Dict:
    """Create analysis plan"""
        
    # Direct prompt to LLM
    response = llm_gemini.invoke(planning_prompt.format(query=query, columns=columns, tool_des= des))
    # Extract JSON from response
    res= clean_and_extract_json(response.content)
    
    plan = json.loads(res)
    add_thought("Planning", plan,thoughts)
    return plan



def analysis(df, extra_tools, query):

    analysis_agent = create_pandas_dataframe_agent(
        llm=llm_gemini,
        df=df,  # Primary dataset
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        extra_tools= extra_tools,
        prefix= "Write proper multiline python script to get solution from the dataframe provided to you. Never define the tools provided to you",
        include_df_in_prompt= True,
        allow_dangerous_code=True
    )
    res= analysis_agent.invoke({
         "input": query
    })
    return res['output']


def synthesize_results(thoughts:List) -> str:
        """Synthesize results"""
        try:
            steps_str = "\n".join([r["step"] for r in thoughts if "step" in r])
            results_str = "\n".join([r["result"] for r in thoughts if "result" in r])
            
            response = llm_gemini.invoke(
                synthesis_prompt.format(
                    steps=steps_str,
                    results=results_str
                )
            )
            synthesis = response.content
            add_thought("Synthesis", synthesis)
            return synthesis
        except Exception as e:
            add_thought("Synthesis Error", str(e))
            print(f"Synthesis error: {str(e)}")
            return "Error synthesizing results"
        


def process_query(df, des, extra_tools, query: str) -> Dict:
        """Process a query using chain of thought reasoning"""
        try:
            # Reset thoughts
            thoughts = []
            add_thought("Initial Query", query,thoughts)
            col= list(df.columns)
            # Plan analysis
            plan = plan_analysis(col, query,thoughts, des)
            
            # Execute steps
            results = []
            for step in plan["steps"]:
                print(step)
                
                result = analysis(df, extra_tools, step)
                results.append({"step": step, "result": result})
            #
            # Synthesize results
            final_result =synthesize_results(results)
            
            return {
                "final_result": final_result,
                #"thoughts": thoughts,
                "success": True
            }
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            add_thought("Error", error_msg, thoughts)
            return {
                "final_result": error_msg,
                "thoughts": thoughts,
                "success": False
            }
        


df= pd.read_csv('./processed_files/Airline_Data/joined_output.csv', encoding='utf-8', encoding_errors='replace')
custom_tool= CustomTool()
extra_tools= [custom_tool]
des= "call to find whether the pilot is new or old. You pass the name and it return True if pilot is new else False."
# result= process_query(df, des, extra_tools, """Count the number of new pilots present?""")
# print(result)

exec('''
new_pilot_count = 0
for pilot_name in df['pilot_name']:
    is_new = custom_tool(pilot_name)
    if is_new:
        new_pilot_count += 1
print(new_pilot_count)
''')