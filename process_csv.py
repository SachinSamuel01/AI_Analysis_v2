import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI 
from langchain.prompts import ChatPromptTemplate
import os
from typing import List, Tuple, Dict
import json

def load_csv(file_path: str) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def analyze_data_quality(df: pd.DataFrame) -> dict:
    """Analyze data quality and return statistics."""
    quality_report = {
        'total_rows': len(df),
        'columns': {},
    }
    
    for column in df.columns:
        column_stats = {
            'missing_values': df[column].isna().sum(),
            'missing_percentage': (df[column].isna().sum() / len(df)) * 100,
            'unique_values': df[column].nunique(),
            'data_type': str(df[column].dtype),
            'sample_values': df[column].dropna().head(3).tolist()
        }
        
        # Additional checks for numeric columns
        if pd.api.types.is_numeric_dtype(df[column]):
            column_stats.update({
                'mean': df[column].mean() if not df[column].empty else None,
                'min': df[column].min() if not df[column].empty else None,
                'max': df[column].max() if not df[column].empty else None,
                'zeros': (df[column] == 0).sum(),
                'negative_values': (df[column] < 0).sum() if not df[column].empty else 0
            })
        
        quality_report['columns'][column] = column_stats
    
    return quality_report

def identify_required_columns(df: pd.DataFrame, llm) -> List[str]:
    """Use OpenAI to identify columns that should not have missing values."""
    column_descriptions = get_column_descriptions(df)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a data quality expert. Your task is to identify columns that should NEVER have missing values in a flight booking dataset.
        These are columns where filling missing values would not make sense (like passenger names, booking IDs, etc.).
        Consider:
        1. Primary identifier columns (booking codes, passenger IDs)
        2. Essential booking information (passenger names, flight numbers)
        3. Critical flight details (departure/arrival times, flight numbers)
        
        Return a JSON array of column names that should never be null.
        Only return the JSON array, no other text."""),
        ("user", f"""
        Here are the columns with sample values:
        {column_descriptions}
        
        Which of these columns should never have missing values?""")
    ])
    
    response = llm.invoke(prompt.format())
    try:
        required_columns = json.loads(response.content)
        print("\nIdentified required columns (rows will be dropped if these contain null values):")
        for col in required_columns:
            print(f"- {col}")
        return required_columns
    except Exception as e:
        print(f"Error parsing LLM response: {str(e)}")
        # Fallback to default critical columns if LLM fails
        default_required = ['passenger_name', 'booking_code', 'flight_number', 
                          'departure_datetime', 'arrival_datetime']
        print("\nUsing default required columns:")
        for col in default_required:
            print(f"- {col}")
        return default_required

def split_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically detect and split columns containing datetime values into separate date and time columns.
    Only splits columns that actually contain both date and time components.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        pd.DataFrame: DataFrame with datetime columns split into date and time
    """
    df_result = df.copy()
    
    # Try to convert each string column to datetime if it contains both date and time
    for col in df_result.columns:
        try:
            # Check if the first non-null value contains both date and time components
            sample_value = df_result[col].dropna().iloc[0]
            if isinstance(sample_value, str) and len(sample_value.split()) > 1:
                # Try converting to datetime
                df_result[col] = pd.to_datetime(df_result[col])
                print(f"Detected datetime format in column: {col}")
        except:
            continue
    
    # Now split the successfully converted datetime columns
    datetime_columns = df_result.select_dtypes(include=['datetime64[ns]']).columns
    
    if not datetime_columns.empty:
        print("\nSplitting datetime columns...")
        
    for col in datetime_columns:
        try:
            # Create new column names
            date_col = f"{col}_date"
            time_col = f"{col}_time"
            
            # Extract date and time components
            df_result[date_col] = df_result[col].dt.date
            df_result[time_col] = df_result[col].dt.time
            
            # Drop the original datetime column
            df_result = df_result.drop(columns=[col])
            
            print(f"- Split '{col}' into '{date_col}' and '{time_col}'")
            
        except Exception as e:
            print(f"Error splitting datetime column '{col}': {str(e)}")
            continue
    
    return df_result

def clean_data(df: pd.DataFrame, quality_report: dict, llm) -> pd.DataFrame:
    """Clean the dataset based on data quality analysis."""
    print("\nCleaning dataset...")
    df_cleaned = df.copy()
    
    # First, identify and handle required columns
    required_columns = identify_required_columns(df_cleaned, llm)
    
    # Drop rows with missing values in required columns
    initial_rows = len(df_cleaned)
    df_cleaned = df_cleaned.dropna(subset=required_columns)
    rows_dropped = initial_rows - len(df_cleaned)
    if rows_dropped > 0:
        print(f"\nDropped {rows_dropped} rows due to missing values in required columns")
    
    # Process remaining columns
    for column, stats in quality_report['columns'].items():
        if column not in required_columns:  # Only process non-required columns
            print(f"\nProcessing column: {column}")
            
            # Handle missing values based on data type and missing percentage
            missing_pct = stats['missing_percentage']
            if missing_pct > 0:
                print(f"- Found {missing_pct:.2f}% missing values")
                
                # If more than 50% values are missing, consider dropping the column
                if missing_pct > 50:
                    print(f"- Dropping column {column} due to high missing value percentage")
                    df_cleaned = df_cleaned.drop(columns=[column])
                    continue
                
                # Handle missing values based on data type
                if pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_bool_dtype(df[column]):
                    # Fill numeric missing values with median
                    median_value = df_cleaned[column].median()
                    df_cleaned[column] = df_cleaned[column].fillna(median_value)
                    print(f"- Filled missing values with median: {median_value}")
                
                elif pd.api.types.is_datetime64_dtype(df[column]):
                    # Forward fill datetime values
                    df_cleaned[column] = df_cleaned[column].fillna(method='ffill')
                    # If still has missing values, backward fill
                    df_cleaned[column] = df_cleaned[column].fillna(method='bfill')
                    print("- Filled missing datetime values using forward/backward fill")
                
                elif pd.api.types.is_bool_dtype(df[column]):
                    # Fill boolean values with mode
                    mode_value = df_cleaned[column].mode().iloc[0]
                    df_cleaned[column] = df_cleaned[column].fillna(mode_value)
                    print(f"- Filled missing boolean values with mode: {mode_value}")
                
                else:
                    # Fill categorical/string missing values with mode
                    mode_value = df_cleaned[column].mode().iloc[0] if not df_cleaned[column].empty else 'Unknown'
                    df_cleaned[column] = df_cleaned[column].fillna(mode_value)
                    print("- Filled missing categorical values with 'Unknown'")
            
            # Handle potential errors in numeric columns (excluding boolean columns)
            if pd.api.types.is_numeric_dtype(df[column]) and not pd.api.types.is_bool_dtype(df[column]):
                # Replace infinite values with NaN
                df_cleaned[column] = df_cleaned[column].replace([np.inf, -np.inf], np.nan)
                
                # Check for outliers using IQR method
                Q1 = df_cleaned[column].quantile(0.25)
                Q3 = df_cleaned[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Replace outliers with bounds
                outliers = df_cleaned[
                    (df_cleaned[column] < lower_bound) | 
                    (df_cleaned[column] > upper_bound)
                ][column].count()
                
                if outliers > 0:
                    print(f"- Found {outliers} outliers")
                    # Convert to float to handle potential integer columns
                    df_cleaned[column] = df_cleaned[column].astype(float)
                    df_cleaned.loc[df_cleaned[column] < lower_bound, column] = lower_bound
                    df_cleaned.loc[df_cleaned[column] > upper_bound, column] = upper_bound
                    print("- Capped outliers to IQR boundaries")
    
    # Remove duplicate rows
    initial_rows = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates()
    duplicates_removed = initial_rows - len(df_cleaned)
    if duplicates_removed > 0:
        print(f"\nRemoved {duplicates_removed} duplicate rows")
    
    return df_cleaned

def get_column_descriptions(df: pd.DataFrame) -> str:
    """Generate a description of DataFrame columns with sample values."""
    descriptions = []
    for col in df.columns:
        sample_values = df[col].dropna().head(3).tolist()
        descriptions.append(f"Column '{col}' with sample values: {sample_values}")
    return "\n".join(descriptions)

def identify_join_columns(main_cols: str, other_cols: str, llm) -> Tuple[str, str]:
    """Use OpenAI to identify matching columns between two DataFrames."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a data analysis expert. Your task is to identify which columns should be used for joining two datasets.
         Analyze the column names and sample values, then return a JSON string with exactly two keys:
         'main_col': the column name from the main dataset
         'other_col': the column name from the other dataset
         Only return the JSON string, no other text."""),
        ("user", f"""
        Main dataset columns:
        {main_cols}
        
        Other dataset columns:
        {other_cols}
        
        Identify the best matching columns for joining these datasets.""")
    ])
    
    response = llm.invoke(prompt.format())
    try:
        result = json.loads(response.content)
        return result['main_col'], result['other_col']
    except Exception as e:
        print(f"Error parsing LLM response: {str(e)}")
        return None, None

def suggest_business_friendly_names(df: pd.DataFrame, llm) -> Dict[str, str]:
    """Use OpenAI to suggest business-friendly column names."""
    column_descriptions = get_column_descriptions(df)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a data analyst expert. Your task is to suggest business-friendly column names 
        that are clear, concise, and easy to query. Follow these rules:
        1. Use snake_case naming convention
        2. Keep names concise but descriptive
        3. Avoid abbreviations unless they are industry standard
        4. Maintain consistency in naming patterns
        5. Include the data type in name when relevant (e.g., is_active, total_amount)
        
        Critical rules for time-related columns:
        1. ALWAYS use full datetime fields as the source of truth for timing. for example
           - scheduled_departure: Full datetime of planned departure
           - scheduled_arrival: Full datetime of planned arrival
        
        3. If additional time-only, columns exist, name them with clear purpose:
           - actual_departure_time: Actual time of depature
           - actual_arrival_time: Actual time of arrival
           - departure_time_local: Local time component only
           - flight_time: Operating time of the flight
           - booking_time: Time when booking was made 

        3. If additional date-only, columns exist, name them with clear purpose:
           - actual_departure_date: Actual date of depature
           - actual_arrival_date: Actual date of arrival
           - departure_date_local: Local date component only
           - flight_date: Operating date of the flight
           - booking_date: Date when booking was made

        4. Never use _datetime suffix as it creates confusion

        Return a JSON string where keys are original column names and values are suggested business-friendly names.
        Only return the JSON string, no other text."""),
        ("user", f"""
        Here are the current columns with sample values:
        {column_descriptions}
        
        Suggest business-friendly names for these columns.""")
    ])
    
    response = llm.invoke(prompt.format())
    try:
        return json.loads(response.content)
    except Exception as e:
        print(f"Error parsing LLM response for column renaming: {str(e)}")
        return {}

def rename_columns(df: pd.DataFrame, llm) -> pd.DataFrame:
    """Rename DataFrame columns to business-friendly names."""
    print("\nSuggesting business-friendly column names...")
    
    # Get new column name suggestions
    new_names = suggest_business_friendly_names(df, llm)
    
    if not new_names:
        print("Could not generate new column names. Keeping original names.")
        return df
    
    # Create a mapping of old to new names and show the changes
    print("\nColumn name changes:")
    for old_name, new_name in new_names.items():
        print(f"{old_name} -> {new_name}")
    
    # Rename the columns
    try:
        df = df.rename(columns=new_names)
        print("\nColumns renamed successfully!")
    except Exception as e:
        print(f"Error renaming columns: {str(e)}")
    
    return df

def join_csv(main_csv: str, other_csvs: List[str], llm):
    """Main function to process and join CSV files."""
    # Load main CSV
    main_df = load_csv(main_csv)
    if main_df is None:
        return
    
    # Process each additional CSV
    join_history = []
    result_df = main_df.copy()
    
    for other_csv in other_csvs:
        print(f"\nProcessing {other_csv}...")
        other_df = load_csv(other_csv)
        if other_df is None:
            continue
            
        # Get column descriptions
        main_cols_desc = get_column_descriptions(result_df)
        other_cols_desc = get_column_descriptions(other_df)
        
        # Identify join columns
        main_col, other_col = identify_join_columns(main_cols_desc, other_cols_desc, llm)
        
        if main_col and other_col:
            # Perform the join
            try:
                result_df = result_df.merge(
                    other_df,
                    left_on=main_col,
                    right_on=other_col,
                    how='left'
                )
                join_history.append({
                    'file': other_csv,
                    'main_column': main_col,
                    'other_column': other_col
                })
                print(f"Successfully joined on columns: {main_col} (main) and {other_col} (other)")
            except Exception as e:
                print(f"Error joining {other_csv}: {str(e)}")
        else:
            print(f"Could not identify join columns for {other_csv}")
        
    # Print join history
    print("\nJoin operations performed:")
    for join in join_history:
        print(f"File: {join['file']}")
        print(f"Joined on: {join['main_column']} (main) = {join['other_column']} (other)\n")
    
    return result_df




def join_csv2(other_csvs: List[str], llm):
    """Main function to process and join CSV files."""
    # Load main CSV
    main_df = load_csv(other_csvs[0])
    if main_df is None:
        return
    
    # Process each additional CSV
    join_history = []
    result_df = main_df.copy()
    
    for i in range(1,len(other_csvs)):
        other_csv= other_csvs[i]
        print(f"\nProcessing {other_csv}...")
        other_df = load_csv(other_csv)
        if other_df is None:
            continue
            
        # Get column descriptions
        main_cols_desc = get_column_descriptions(result_df)
        other_cols_desc = get_column_descriptions(other_df)
        
        # Identify join columns
        main_col, other_col = identify_join_columns(main_cols_desc, other_cols_desc, llm)
        
        if main_col and other_col:
            # Perform the join
            try:
                result_df = result_df.merge(
                    other_df,
                    left_on=main_col,
                    right_on=other_col,
                    how='left'
                )
                join_history.append({
                    'file': other_csv,
                    'main_column': main_col,
                    'other_column': other_col
                })
                print(f"Successfully joined on columns: {main_col} (main) and {other_col} (other)")
            except Exception as e:
                print(f"Error joining {other_csv}: {str(e)}")
        else:
            print(f"Could not identify join columns for {other_csv}")
        
    # Print join history
    print("\nJoin operations performed:")
    for join in join_history:
        print(f"File: {join['file']}")
        print(f"Joined on: {join['main_column']} (main) = {join['other_column']} (other)\n")
    
    return result_df

def clean_data_begin(uploaded_path, saved_path):
    # Example usage
    OPENAI_API_KEY = os.getenv("OPENAI_KEY")
    
    OTHER_CSVS = [
        os.path.join(uploaded_path,i) for i in os.listdir(uploaded_path)
    ]
    
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    result_df = join_csv2(OTHER_CSVS, llm)
    result_df = rename_columns(result_df, llm)

    # Analyze and clean main dataset
    print("\nAnalyzing main dataset quality...")
    quality_report = analyze_data_quality(result_df)
    result_df = clean_data(result_df, quality_report, llm)

    # Split datetime columns
    result_df = split_datetime_columns(result_df)
    print("Final split columns")
    print(result_df.columns)

    output_path = f'{saved_path}/joined_output.csv'
    result_df.to_csv(output_path, index=False)
    return f"final output saved to: {output_path}"