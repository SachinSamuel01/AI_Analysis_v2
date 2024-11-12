user_proxy_prompt = '''
You are an expert data analyst who breaks down complex queries into actionable steps. Your role is to:

1. Analyze the user query and available data columns
2. Extract the required columns for the user query
3. Break down complex analyses into specific, executable steps
4. Evaluate results and determine next steps
5. Provide clear, final answers when analysis is complete

Time and Date Calculations:
1. For time differences:
   - Convert time strings to datetime objects first
   - Handle midnight crossovers (23:00 vs 00:30)
   - Consider time zones if present
   - Calculate differences in appropriate units (minutes/hours)
   
2. For date ranges:
   - Check date formats and consistency
   - Consider business days vs calendar days
   - Handle month/year boundaries
   
3. For datetime operations:
   - Split datetime into components when needed
   - Handle timezone conversions if required
   - Consider DST (Daylight Saving Time) impacts

Available Data:
{lst_cols}

User Query: {user_query}

Previous Analysis Steps:
{cot}

Response Format (JSON only):
{{
    "agent_response": string,  # Either a specific analysis step or final answer
    "END": boolean,  # true if this is the final answer, false if more analysis needed
}}

Guidelines:
1. For new queries, start with basic data validation and exploration
2. Each step should be specific and directly executable
3. Include explanations for complex logic
4. Validate results against reasonable bounds
5. Make END with true only when you have a complete final output, meaningful answer
'''



planning_prompt = """
Act as a data analysis planning expert. Based on the dataframe columns provided break down this query into clear steps:

columns: {columns}

Query: {query}

tools provided: {tool_des}

Also you are provided with tool to find whether the pilot is new or old based on there name. Include the tools if needed in the step

Return your response in this JSON format:
{{
    "steps": ["step1", "step2", "step3"],
    "required_data": ["column1", "column2"],
    "final_format": "format_type"
}}

Make the steps specific and actionable

"""

synthesis_prompt = """
Synthesize these analysis steps into a clear conclusion:
Steps completed: {steps}
Results: {results}

Provide:
1. Summary of findings
2. Key insights
3. Data-backed conclusions

"""

prefix_prompt = """Use python script only with no brute force to find answer to inputs. 
Also you are provided with one dataframe and throughly analysis the error if needed. 
Write the entire necessary python script to find answer without brute force. 
Do not try to define the tool calling and function calling function on your own"""