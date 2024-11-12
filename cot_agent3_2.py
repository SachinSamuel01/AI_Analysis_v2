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
from langchain.tools import BaseTool, Tool
from langchain_experimental.tools import PythonAstREPLTool 


load_dotenv()

def check_if_new_pilot(name: str) -> bool:
    """Check if pilot is new based on name"""
    return False if 'a' == name[-1] else True



class ChainOfThoughtAgent:
    def __init__(self, llm, base_path: str, csv_files: List[str]):
        """
        Initialize the Chain of Thought agent with multiple specialized sub-agents.
        
        Args:
            llm: Language model instance
            base_path: Base path to CSV files
            csv_files: List of CSV file names
        """
        self.llm = llm
        self.base_path = base_path
        self.csv_files = csv_files
        self.thoughts = []
        
        self.load_dataframes()
        self.initialize_agents()
        
    def load_dataframes(self):
        """Load and preprocess all CSV files"""
        self.dataframes = {}
        for file in self.csv_files:
            try:
                path = os.path.join(self.base_path, file)
                df = pd.read_csv(path, encoding='utf-8', encoding_errors='replace')
                # Clean column names and handle basic preprocessing
                df.columns = df.columns.str.strip().str.lower()
                self.dataframes[file] = df
                print(f"Loaded {file} with shape {df.shape}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                
    def initialize_agents(self):
        """Initialize specialized agents for different analysis tasks"""

        # Create a custom namespace for the Python REPL
        namespace = {
            'df': list(self.dataframes.values())[0],  # Primary dataset
            'pd': pd,
            'pilot_checker': check_if_new_pilot
        }
        
        # Create Python REPL tool with the custom namespace
        python_repl = PythonAstREPLTool(locals=namespace)
        
        tools = [
            Tool(
                name="python_repl_ast",
                #name= "pilot_checker",
                func=python_repl.run,
                description="""A Python shell. Use this to execute python commands. Input should be a valid python command. 
                When you want to analyze pilots, you can directly use the pilot_checker function which is already imported.
                For example: result = df[df['pilot_name'].apply(pilot_checker)]"""
            )
        ]

        # Planning agent to break down complex queries
        self.planning_prompt = """
        Act as a data analysis planning expert. Break down this query into clear steps:
        Query: {query}
        
        Return your response in this JSON format:
        {{
            "steps": ["step1", "step2", "step3"],
            "required_data": ["column1", "column2"],
            "final_format": "format_type"
        }}
        
        Make the steps specific and actionable.
        """
        
        # Data analysis agent for executing individual steps
        self.analysis_agent = create_pandas_dataframe_agent(
            llm=self.llm,
            df=list(self.dataframes.values())[0],  # Primary dataset
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            extra_tools=tools,
            prefix= """You have access to a pandas dataframe and a tool called 'pilot_checker'.
                     To check if a pilot is new, use: pilot_checker
                     Use Python pandas operations for data analysis and the pilot_checker tool when needed.""",
            include_df_in_prompt= True,
            allow_dangerous_code=True
        )
        
        # Results synthesis agent
        self.synthesis_prompt = """
        Synthesize these analysis steps into a clear conclusion:
        Steps completed: {steps}
        Results: {results}
        
        Provide:
        1. Summary of findings
        2. Key insights
        3. Data-backed conclusions
        4. provided with tool, call to find whether the pilot is new or old based on there name

        """

    def add_thought(self, step: str, result: Any):
        """Record a step in the chain of thought"""
        self.thoughts.append({
            "step": step,
            "result": str(result),
            "timestamp": pd.Timestamp.now()
        })

    def plan_analysis(self, query: str) -> Dict:
        """Create analysis plan"""
        try:
            # Direct prompt to LLM
            response = self.llm.invoke(self.planning_prompt.format(query=query))
            # Extract JSON from response
            plan = json.loads(response.content)
            self.add_thought("Planning", plan)
            return plan
        except Exception as e:
            self.add_thought("Planning Error", str(e))
            print(f"Planning error: {str(e)}")
            # Return a basic plan for simple queries
            return {
                "steps": ["Count records by airline"],
                "required_data": ["airline_name"],
                "final_format": "dataframe"
            }

    def execute_step(self, step: str) -> pd.DataFrame:
        """Execute a single analysis step"""
        try:
            result = self.analysis_agent.invoke({
                "input": f"Execute this step and return results as a DataFrame: {step}"
            })
            
            # Handle different result types
            if isinstance(result, dict) and 'output' in result:
                result = result['output']
            if not isinstance(result, pd.DataFrame):
                if isinstance(result, str):
                    # Try to convert string table to DataFrame
                    try:
                        result = pd.read_csv(pd.StringIO(result), sep='|')
                    except:
                        result = pd.DataFrame({'result': [result]})
                else:
                    result = pd.DataFrame({'result': [str(result)]})
                    
            self.add_thought(f"Executed: {step}", result)
            return result
        except Exception as e:
            self.add_thought(f"Step Error: {step}", str(e))
            print(f"Execution error: {str(e)}")
            return pd.DataFrame({'error': [str(e)]})

    def synthesize_results(self, results: List[Dict]) -> str:
        """Synthesize results"""
        try:
            steps_str = "\n".join([r["step"] for r in self.thoughts if "step" in r])
            results_str = "\n".join([r["result"] for r in self.thoughts if "result" in r])
            
            response = self.llm.invoke(
                self.synthesis_prompt.format(
                    steps=steps_str,
                    results=results_str
                )
            )
            synthesis = response.content
            self.add_thought("Synthesis", synthesis)
            return synthesis
        except Exception as e:
            self.add_thought("Synthesis Error", str(e))
            print(f"Synthesis error: {str(e)}")
            return "Error synthesizing results"

    def process_query(self, query: str) -> Dict:
        """Process a query using chain of thought reasoning"""
        try:
            # Reset thoughts
            self.thoughts = []
            self.add_thought("Initial Query", query)
            
            # Plan analysis
            plan = self.plan_analysis(query)
            
            # Execute steps
            results = []
            for step in plan["steps"]:
                result = self.execute_step(step)
                results.append({"step": step, "result": result})
            
            # Synthesize results
            final_result = self.synthesize_results(results)
            
            return {
                "final_result": final_result,
                "thoughts": self.thoughts,
                "success": True
            }
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            self.add_thought("Error", error_msg)
            return {
                "final_result": error_msg,
                "thoughts": self.thoughts,
                "success": False
            }

# Example usage:
if __name__ == "__main__":
    
    
    # llm = ChatOpenAI(
    #     model="gpt-4",
    #     temperature=0,
    #     top_p=0.1,
    #     api_key=os.getenv("OPENAI_KEY"),
    # )
    
    # llm= ChatGoogleGenerativeAI(
    #     model= "gemini-1.5-flash",
    #     temperature= 0,
    #     top_p= 0.1,
    #     api_key= os.getenv('GEMINI_API_KEY')
    # )

    llm = ChatOpenAI(
        model="gpt-4",
        temperature=0,
        top_p=0.1,
        api_key=os.getenv("OPENAI_KEY"),
    )

    # Example path and files
    base_path = "./processed_files/Airline_Data"
    csv_files = ["joined_output.csv"]
    
    agent = ChainOfThoughtAgent(llm, base_path, csv_files)
    
    # Test with a complex query
    query = """Tell me how many new pilots are there in the data?"""
    
    result = agent.process_query(query)
    
    print("\nAnalysis Results:")
    print("Final Result:", result["final_result"])
    print("\nChain of Thought Process:")
    for thought in result["thoughts"]:
        print(f"\nStep: {thought['step']}")
        print(f"Result: {thought['result'][:200]}...")  # Truncate long results