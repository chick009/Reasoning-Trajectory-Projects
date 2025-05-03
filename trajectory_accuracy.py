import pandas as pd
import numpy as np
from openai import OpenAI
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import time
import re
import os
import sys
from dotenv import load_dotenv
from datasets import load_dataset

# Load environment variables from .env file if it exists
load_dotenv()

# Set up logging
def log(message):
    """Print a log message with timestamp and flush to ensure output is shown immediately"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()  # Force output to display immediately

# Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    log("Error: DEEPSEEK_API_KEY environment variable not found in .env file")
    sys.exit(1)

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-reasoner"
EVALUATOR_MODEL_NAME = "deepseek-chat"
MAX_PROBLEMS_PER_DATASET = 30 # Limit problems for testing

# Datasets to process
DATASETS_TO_PROCESS = [
    {
        "name": "AIME2024", 
        "hf_id": "Maxwell-Jia/AIME_2024", 
        "config": None, 
        "split": "train", 
        "problem_field": "Problem", 
        "answer_field": "Answer"
    },
    {
        "name": "AIME2023_I", 
        "hf_id": "MathArena/aime_2023_I", 
        "config": None, 
        "split": "train", 
        "problem_field": "problem", 
        "answer_field": "answer"
    },
    {
        "name": "AIME2023_II", 
        "hf_id": "MathArena/aime_2023_II", # Assuming this exists and has similar structure
        "config": None, 
        "split": "train", 
        "problem_field": "problem", 
        "answer_field": "answer"
    },
]

def load_aime_dataset(dataset_info):
    """Load specified AIME dataset from Hugging Face."""
    hf_id = dataset_info["hf_id"]
    config = dataset_info["config"]
    split = dataset_info["split"]
    log(f"Loading {dataset_info['name']} dataset ({hf_id})...")
    try:
        if config:
            dataset = load_dataset(hf_id, config, split=split)
        else:
            dataset = load_dataset(hf_id, split=split)
        log(f"Dataset {dataset_info['name']} loaded successfully.")
        return dataset
    except Exception as e:
        log(f"Error loading dataset {hf_id}: {e}")
        return None

def get_reasoning_trajectory(problem):
    """Use DeepSeek Reasoner to generate reasoning trajectory for a problem."""
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    
    prompt = f"""Solve this mathematics problem step by step, showing all your work:
    
{problem}
    
Provide your final answer as a single number at the end in the format \\boxed{{your answer}}."""
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a mathematical problem solver. Show all your reasoning steps clearly."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        
        reasoning = response.choices[0].message.reasoning_content
        final_answer = response.choices[0].message.content
        
        return {
            "reasoning": reasoning,
            "final_answer": final_answer
        }
    except Exception as e:
        log(f"Error getting reasoning trajectory: {e}")
        return {
            "reasoning": "",
            "final_answer": ""
        }

def evaluate_reasoning_quality_with_model(problem, reasoning):
    """
    Evaluate the quality of reasoning on a scale from 1-10 using deepseek-chat.
    """
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    
    prompt = f"""Your task is to evaluate the quality of a mathematical problem-solving reasoning trajectory.
    
Problem:
{problem}

Reasoning trajectory:
{reasoning}

Rate the quality of this reasoning trajectory on a scale from 1 to 10 (10 being the highest quality).
Consider the following factors in your rating:
1. Clarity of explanation
2. Logical coherence of steps
3. Correctness of intermediate steps
4. Appropriate use of mathematical notation
5. Completeness of the solution

Provide a single integer rating from 1-10, with no additional text."""
    
    try:
        response = client.chat.completions.create(
            model=EVALUATOR_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert mathematics evaluator. Your job is to rate the quality of mathematical reasoning."},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        
        rating_text = response.choices[0].message.content.strip()
        
        # Extract the numeric rating
        match = re.search(r'\b([1-9]|10)\b', rating_text)
        if match:
            rating = int(match.group(1))
            return rating
        else:
            log(f"Could not extract rating from response: {rating_text}")
            return 5  # Default to middle rating if extraction fails
    except Exception as e:
        log(f"Error evaluating reasoning quality: {e}")
        return 5  # Default to middle rating

def extract_numeric_answer(answer_text):
    """Extract the numeric answer from the model's response."""
    # Try to find a boxed answer
    match = re.search(r'\\boxed{(\d+)}', answer_text)
    if match:
        return match.group(1)
    
    # Try to find a number at the end of the text
    match = re.search(r'(\d+)(?:\s*\.)?$', answer_text.strip())
    if match:
        return match.group(1)
    
    # Try to find any number in the text
    match = re.search(r'\b(\d+)\b', answer_text)
    if match:
        return match.group(1)
    
    return None

def check_answer_correctness(predicted_answer, actual_answer):
    """Check if the predicted answer matches the actual answer."""
    # Clean up answers for comparison
    if predicted_answer is None:
        return False
    
    predicted_clean = str(predicted_answer).strip()
    actual_clean = str(actual_answer).strip()
    
    return predicted_clean == actual_clean

def analyze_correlation(df, dataset_name):
    """Analyze correlation between reasoning quality and answer correctness."""
    if len(df) < 2:
        log(f"Cannot analyze correlation for {dataset_name}, less than 2 data points.")
        return
        
    correlation = np.corrcoef(df['reasoning_quality'], df['is_correct'])[0, 1]
    
    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['reasoning_quality'], df['is_correct'], alpha=0.6)
    plt.title(f'Correlation for {dataset_name}: {correlation:.2f}')
    plt.xlabel('Reasoning Quality (1-10)')
    plt.ylabel('Answer Correctness (0=Incorrect, 1=Correct)')
    plt.yticks([0, 1], ['Incorrect', 'Correct'])
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add best fit line if possible
    if df['reasoning_quality'].nunique() > 1 and df['is_correct'].nunique() > 1:
        x = df['reasoning_quality'].values.reshape(-1, 1)
        y = df['is_correct'].values
        try:
            model = LogisticRegression()
            model.fit(x, y)
            x_range = np.linspace(df['reasoning_quality'].min(), df['reasoning_quality'].max(), 100).reshape(-1, 1)
            y_proba = model.predict_proba(x_range)[:, 1]
            plt.plot(x_range, y_proba, color='red', linestyle='--')
        except ValueError as e:
            log(f"Could not fit logistic regression for {dataset_name}: {e}")
    
    plot_filename = f'correlation_analysis_{dataset_name}.png'
    plt.savefig(plot_filename)
    plt.close()
    log(f"Correlation plot saved as {plot_filename}")
    
    return correlation

def process_dataset(dataset_info):
    """Loads, processes, and analyzes a single dataset."""
    log(f"\n--- Starting processing for dataset: {dataset_info['name']} ---")
    dataset = load_aime_dataset(dataset_info)
    if dataset is None:
        log(f"Skipping dataset {dataset_info['name']} due to loading error.")
        return
    
    df_dataset = pd.DataFrame(dataset)
    results = []
    
    num_to_process = min(MAX_PROBLEMS_PER_DATASET, len(df_dataset))
    log(f"Processing {num_to_process} problems from {dataset_info['name']}...")
    
    problem_field = dataset_info['problem_field']
    answer_field = dataset_info['answer_field']

    # Process each problem (limit for testing)
    for i, row in df_dataset[:num_to_process].iterrows():
        problem_id = f"{dataset_info['name']}-{i+1}"
        
        # Check if required fields exist
        if problem_field not in row or answer_field not in row:
            log(f"Warning: Missing field '{problem_field if problem_field not in row else answer_field}' in row {i} of {dataset_info['name']}. Skipping.")
            continue
            
        problem = row[problem_field]
        actual_answer = row[answer_field]
        
        log(f"-- Processing problem {problem_id} ({i+1}/{num_to_process}) --")
        log(f"Problem: {problem[:150]}{'...' if len(problem) > 150 else ''}")
        
        # Get reasoning trajectory
        trajectory = get_reasoning_trajectory(problem)
        reasoning = trajectory["reasoning"]
        final_answer = trajectory["final_answer"]
        
        log(f"Generated reasoning length: {len(reasoning) if reasoning else 0}")
        
        # Extract numeric answer
        predicted_answer = extract_numeric_answer(final_answer)
        log(f"Extracted answer: {predicted_answer}")
        
        # Evaluate reasoning quality using deepseek-chat
        log(f"Evaluating reasoning quality for problem {problem_id}...")
        quality = evaluate_reasoning_quality_with_model(problem, reasoning)
        log(f"Quality rating: {quality}/10")
        
        # Check answer correctness
        is_correct = check_answer_correctness(predicted_answer, actual_answer)
        log(f"Correct: {is_correct}")
        
        # Store results
        results.append({
            "problem_id": problem_id,
            "problem": problem,
            "reasoning": reasoning,
            "final_answer": final_answer,
            "predicted_answer": predicted_answer,
            "actual_answer": actual_answer,
            "reasoning_quality": quality,
            "is_correct": int(is_correct)
        })
        
        # Avoid rate limiting
        log(f"Waiting 2 seconds before next problem...")
        time.sleep(2)
    
    if not results:
        log(f"No results generated for dataset {dataset_info['name']}. Cannot proceed with analysis.")
        return
        
    # Create dataframe from results
    df = pd.DataFrame(results)
    
    # Save results to CSV
    results_file = f"results_{dataset_info['name']}.csv"
    df.to_csv(results_file, index=False)
    log(f"Results for {dataset_info['name']} saved to {results_file}")
    
    # Analyze correlation
    correlation = analyze_correlation(df, dataset_info['name'])
    log(f"Correlation for {dataset_info['name']}: {correlation:.2f}")
    
    # Print summary statistics
    correct_count = df['is_correct'].sum()
    total_count = len(df)
    accuracy = (correct_count / total_count) if total_count > 0 else 0
    avg_quality = df['reasoning_quality'].mean()
    
    log(f"\nSummary Statistics for {dataset_info['name']}:")
    log(f"Total problems analyzed: {total_count}")
    log(f"Correct answers: {correct_count} ({accuracy:.2%})")
    log(f"Average reasoning quality: {avg_quality:.2f}/10")
    
    # Group by quality score and calculate accuracy
    if total_count > 0:
        quality_groups = df.groupby('reasoning_quality')['is_correct'].agg(['count', 'mean'])
        log("\nAccuracy by Reasoning Quality:")
        log(quality_groups)
    
    log(f"--- Finished processing for dataset: {dataset_info['name']} ---")

def main():
    log("Starting AIME evaluation process for multiple datasets...")
    for dataset_info in DATASETS_TO_PROCESS:
        process_dataset(dataset_info)
    log("All datasets processed!")

if __name__ == "__main__":
    main() 