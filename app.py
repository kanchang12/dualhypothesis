from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from openai import OpenAI
import os
import time

app = Flask(__name__)

# Setup
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
gemini = genai.GenerativeModel('gemini-2.0-flash-exp')
openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

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
    cost = 0  # Gemini 2.5 Flash is free
    
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
    
    gemini_cost = 0  # Free
    openai_cost = (openai_in * 0.15 + openai_out * 0.60) / 1_000_000
    
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
        'gemini_cost': gemini_cost,
        'openai_cost': round(openai_cost, 6),
        'total_cost': round(openai_cost, 6)
    })

if __name__ == '__main__':
    app.run(port=8080, debug=True)
