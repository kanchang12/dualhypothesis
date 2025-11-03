from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from openai import OpenAI
import os
import time

app = Flask(__name__)

# Setup
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
gemini = genai.GenerativeModel('gemini-pro')  # PAID MODEL
openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# Pricing per 1M tokens
GEMINI_INPUT_COST = 0.50 / 1_000_000    # $0.50 per 1M input
GEMINI_OUTPUT_COST = 1.50 / 1_000_000   # $1.50 per 1M output
OPENAI_INPUT_COST = 0.150 / 1_000_000
OPENAI_OUTPUT_COST = 0.600 / 1_000_000

def count_tokens(text):
    return int(len(text.split()) * 1.3)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/gemini', methods=['POST'])
def gemini_only():
    user_prompt = request.json['prompt']
    start = time.time()
    
    # Proper prompt for Gemini to generate code
    full_prompt = f"""You are a code generator. Generate clean, working code based on the requirement below.

Requirement: {user_prompt}

Generate only the code, no explanations."""
    
    response = gemini.generate_content(full_prompt)
    code = response.text
    
    elapsed = time.time() - start
    in_tokens = count_tokens(full_prompt)
    out_tokens = count_tokens(code)
    cost = (in_tokens * GEMINI_INPUT_COST) + (out_tokens * GEMINI_OUTPUT_COST)
    
    return jsonify({
        'code': code,
        'time': round(elapsed, 2),
        'gemini_in': in_tokens,
        'gemini_out': out_tokens,
        'cost': cost
    })

@app.route('/api/dual', methods=['POST'])
def gemini_plus_openai():
    user_prompt = request.json['prompt']
    start = time.time()
    
    # Step 1: Gemini generates code with proper prompt
    generation_prompt = f"""You are a code generator. Generate clean, working code based on the requirement below.

Requirement: {user_prompt}

Generate only the code, no explanations."""
    
    response = gemini.generate_content(generation_prompt)
    code = response.text
    gemini_time = time.time() - start
    
    # Step 2: OpenAI validates with proper prompt
    validation_prompt = f"""You are a code validator. Your ONLY task is to check if the generated code matches the requirement.

REQUIREMENT:
{user_prompt}

GENERATED CODE:
{code}

Check line by line:
1. Does the code fulfill the requirement?
2. Is the code complete (no missing parts)?
3. Are there obvious errors?

Respond ONLY with JSON:
{{"valid": true or false, "reason": "brief explanation"}}"""
    
    openai_start = time.time()
    validation = openai_client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': validation_prompt}],
        temperature=0
    )
    openai_time = time.time() - openai_start
    
    validation_text = validation.choices[0].message.content
    
    elapsed = time.time() - start
    gemini_in = count_tokens(generation_prompt)
    gemini_out = count_tokens(code)
    openai_in = count_tokens(validation_prompt)
    openai_out = count_tokens(validation_text)
    
    gemini_cost = (gemini_in * GEMINI_INPUT_COST) + (gemini_out * GEMINI_OUTPUT_COST)
    openai_cost = (openai_in * OPENAI_INPUT_COST) + (openai_out * OPENAI_OUTPUT_COST)
    
    return jsonify({
        'code': code,
        'validation': validation_text,
        'time': round(elapsed, 2),
        'gemini_time': round(gemini_time, 2),
        'openai_time': round(openai_time, 2),
        'gemini_in': gemini_in,
        'gemini_out': gemini_out,
        'openai_in': openai_in,
        'openai_out': openai_out,
        'gemini_cost': round(gemini_cost, 6),
        'openai_cost': round(openai_cost, 6),
        'total_cost': round(gemini_cost + openai_cost, 6)
    })

@app.route('/api/prompts', methods=['GET'])
def get_prompts():
    """Return all 50 test prompts"""
    prompts = [
        "Write a Python function to check if a number is even",
        "Write a function to reverse a string",
        "Write a function to find max of three numbers",
        "Write a function to calculate factorial",
        "Write a function to check if string is palindrome",
        "Write a function to count vowels in a string",
        "Write a function to convert Celsius to Fahrenheit",
        "Write a function to check if year is leap year",
        "Write a function to find sum of list",
        "Write a function to remove duplicates from list",
        "Write a function to check if string has only digits",
        "Write a function to find second largest in list",
        "Write a function to generate random number in range",
        "Write a function to capitalize first letter",
        "Write a function to calculate area of circle",
        "Write a Python class for a library system with add and search",
        "Write a Flask API endpoint for user login with JWT",
        "Write a function to parse CSV and return dictionaries",
        "Write a debounce function in JavaScript",
        "Write a Python decorator to measure execution time",
        "Write a function to implement binary search",
        "Write a function to validate email with regex",
        "Write a Python class for banking with deposit and withdraw",
        "Write a function to convert JSON to XML",
        "Write a function to compress files with gzip",
        "Write a Python function to send email via SMTP",
        "Write a function to parse JWT tokens",
        "Write a cache class with LRU eviction",
        "Write a function to calculate Levenshtein distance",
        "Write a function to implement rate limiting",
        "Write a function to connect to PostgreSQL safely",
        "Write a Python script to scrape website prices",
        "Write a function for infinite scroll pagination",
        "Write a function to monitor API health",
        "Write a function for drag and drop in JavaScript",
        "Write a merge sort implementation in Python",
        "Write a Flask API with auth and CRUD using SQLAlchemy",
        "Write a thread-safe singleton class in Python",
        "Write a function to solve N-Queens using backtracking",
        "Write Dijkstra's shortest path with priority queue",
        "Write a basic blockchain with proof of work",
        "Write a trie data structure with autocomplete",
        "Write a connection pool with health checks",
        "Write A* pathfinding for grid navigation",
        "Write a WebSocket chat server with rooms",
        "Write a distributed task queue with Redis",
        "Write a virtual DOM diffing algorithm",
        "Write a microservice with Docker compose",
        "Write a React component with real-time editing",
        "Write a CI/CD pipeline in GitHub Actions YAML"
    ]
    return jsonify({'prompts': prompts})

if __name__ == '__main__':
    app.run(port=8080, debug=True)
