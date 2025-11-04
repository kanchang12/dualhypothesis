from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from openai import OpenAI
import os
import time
import json
import sys
from io import StringIO
import traceback

app = Flask(__name__)

# Setup
genai.configure(api_key=os.environ.get('GEMINI_API_KEY'))
gemini = genai.GenerativeModel('gemini-2.5-flash')
openai_client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# Pricing per 1M tokens (Paid Tier)
GEMINI_INPUT_COST = 0.30 / 1_000_000     # $0.30 per 1M input tokens
GEMINI_OUTPUT_COST = 2.50 / 1_000_000    # $2.50 per 1M output tokens (including thinking)
OPENAI_INPUT_COST = 0.150 / 1_000_000    # GPT-4o-mini
OPENAI_OUTPUT_COST = 0.600 / 1_000_000   # GPT-4o-mini

def count_tokens(text):
    return int(len(text.split()) * 1.3)

def test_code(code, test_cases):
    """Execute code and run test cases - STRICT validation"""
    try:
        # Create isolated namespace
        namespace = {}
        
        # Execute the code - if this fails, code is broken
        exec(code, namespace)
        
        # Run each test case
        for test in test_cases:
            func_name = test['function']
            inputs = test['input']
            expected = test['expected']
            
            # Function must exist
            if func_name not in namespace:
                return False, f"Function '{func_name}' not found in code"
            
            func = namespace[func_name]
            
            # Function must be callable
            if not callable(func):
                return False, f"'{func_name}' is not a function"
            
            # Call function with inputs
            try:
                if isinstance(inputs, list):
                    result = func(*inputs)
                else:
                    result = func(inputs)
            except Exception as e:
                return False, f"Function crashed: {str(e)}"
            
            # STRICT comparison - must match exactly
            if result != expected:
                return False, f"Test failed: {func_name}({inputs}) returned {result}, expected {expected}"
        
        return True, "All tests passed"
        
    except SyntaxError as e:
        return False, f"Syntax error in code: {str(e)}"
    except Exception as e:
        return False, f"Code execution error: {str(e)}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/path1', methods=['POST'])
def path1():
    """Path 1: Gemini → Unit Test → Retry until pass"""
    data = request.json
    prompt_text = data['prompt']
    test_cases = data['test_cases']
    conversation_history = data.get('conversation_history', [])
    
    attempts = []
    passed = False
    max_attempts = 10
    final_code = ""
    
    # Build full prompt with history
    if conversation_history:
        full_prompt = "\n\n".join(conversation_history) + "\n\n" + prompt_text
    else:
        full_prompt = prompt_text
    
    for attempt in range(max_attempts):
        start = time.time()
        
        # Generate with Gemini
        response = gemini.generate_content(full_prompt)
        code = response.text.strip()
        
        # Clean markdown if present
        if '```python' in code:
            code = code.split('```python')[1].split('```')[0].strip()
        elif '```' in code:
            code = code.split('```')[1].split('```')[0].strip()
        
        elapsed = time.time() - start
        
        # Calculate costs
        in_tokens = count_tokens(full_prompt)
        out_tokens = count_tokens(code)
        cost = (in_tokens * GEMINI_INPUT_COST) + (out_tokens * GEMINI_OUTPUT_COST)
        
        # Test the code
        test_passed, test_message = test_code(code, test_cases)
        
        attempts.append({
            'attempt': attempt + 1,
            'time': round(elapsed, 2),
            'tokens_in': in_tokens,
            'tokens_out': out_tokens,
            'cost': round(cost, 6),
            'test_passed': test_passed,
            'test_message': test_message
        })
        
        if test_passed:
            passed = True
            final_code = code
            break
        
        final_code = code
    
    # Calculate totals
    total_time = sum(a['time'] for a in attempts)
    total_tokens = sum(a['tokens_in'] + a['tokens_out'] for a in attempts)
    total_cost = sum(a['cost'] for a in attempts)
    
    return jsonify({
        'success': True,
        'passed': passed,
        'attempts': len(attempts),
        'total_time': round(total_time, 2),
        'total_tokens': total_tokens,
        'total_cost': round(total_cost, 6),
        'details': attempts,
        'final_code': final_code
    })

@app.route('/api/path2', methods=['POST'])
def path2():
    """Path 2: Gemini → OpenAI Validation → Unit Test → Retry if fail"""
    data = request.json
    prompt_text = data['prompt']
    test_cases = data['test_cases']
    conversation_history = data.get('conversation_history', [])
    
    attempts = []
    passed = False
    max_attempts = 10
    final_code = ""
    
    # Build full prompt with history
    if conversation_history:
        full_prompt = "\n\n".join(conversation_history) + "\n\n" + prompt_text
    else:
        full_prompt = prompt_text
    
    for attempt in range(max_attempts):
        # Step 1: Generate with Gemini
        gemini_start = time.time()
        response = gemini.generate_content(full_prompt)
        code = response.text.strip()
        
        # Clean markdown
        if '```python' in code:
            code = code.split('```python')[1].split('```')[0].strip()
        elif '```' in code:
            code = code.split('```')[1].split('```')[0].strip()
        
        gemini_time = time.time() - gemini_start
        gemini_in = count_tokens(full_prompt)
        gemini_out = count_tokens(code)
        gemini_cost = (gemini_in * GEMINI_INPUT_COST) + (gemini_out * GEMINI_OUTPUT_COST)
        
        # Step 2: OpenAI validates
        validation_prompt = f"""Check if this code is correct for the requirement.

REQUIREMENT: {prompt_text}

CODE:
{code}

Respond ONLY with JSON:
{{"valid": true or false, "reason": "brief"}}"""
        
        openai_start = time.time()
        validation = openai_client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{'role': 'user', 'content': validation_prompt}],
            temperature=0
        )
        openai_time = time.time() - openai_start
        
        validation_text = validation.choices[0].message.content
        openai_in = count_tokens(validation_prompt)
        openai_out = count_tokens(validation_text)
        openai_cost = (openai_in * OPENAI_INPUT_COST) + (openai_out * OPENAI_OUTPUT_COST)
        
        # Parse validation
        try:
            if '```json' in validation_text:
                validation_text = validation_text.split('```json')[1].split('```')[0].strip()
            validation_result = json.loads(validation_text)
            openai_approved = validation_result.get('valid', False)
        except:
            openai_approved = False
        
        # If OpenAI rejects, retry generation (don't run unit test)
        if not openai_approved:
            attempts.append({
                'attempt': attempt + 1,
                'time': round(gemini_time + openai_time, 2),
                'gemini_tokens': gemini_in + gemini_out,
                'openai_tokens': openai_in + openai_out,
                'cost': round(gemini_cost + openai_cost, 6),
                'openai_approved': False,
                'test_passed': False,
                'test_message': 'OpenAI rejected - did not run unit test'
            })
            continue
        
        # Step 3: Run unit test (only if OpenAI approved)
        test_passed, test_message = test_code(code, test_cases)
        
        attempts.append({
            'attempt': attempt + 1,
            'time': round(gemini_time + openai_time, 2),
            'gemini_tokens': gemini_in + gemini_out,
            'openai_tokens': openai_in + openai_out,
            'cost': round(gemini_cost + openai_cost, 6),
            'openai_approved': True,
            'test_passed': test_passed,
            'test_message': test_message
        })
        
        if test_passed:
            passed = True
            final_code = code
            break
        
        final_code = code
    
    # Calculate totals
    total_time = sum(a['time'] for a in attempts)
    total_tokens = sum(a['gemini_tokens'] + a['openai_tokens'] for a in attempts)
    total_cost = sum(a['cost'] for a in attempts)
    
    return jsonify({
        'success': True,
        'passed': passed,
        'attempts': len(attempts),
        'total_time': round(total_time, 2),
        'total_tokens': total_tokens,
        'total_cost': round(total_cost, 6),
        'details': attempts,
        'final_code': final_code
    })

@app.route('/api/prompts', methods=['GET'])
def get_prompts():
    """Return 50 prompts: 10 base problems with 5 MESSY human-style modifications each"""
    prompts = [
        # PROBLEM 1: Even checker
        {
            'prompt': 'write a function to check if a number is even',
            'test_cases': [
                {'function': 'is_even', 'input': 4, 'expected': True},
                {'function': 'is_even', 'input': 7, 'expected': False}
            ]
        },
        {
            'prompt': 'no wait i need it called is_even, and also what about zero',
            'test_cases': [
                {'function': 'is_even', 'input': 0, 'expected': True},
                {'function': 'is_even', 'input': 4, 'expected': True}
            ]
        },
        {
            'prompt': 'actually can you make it return 1 or 0 instead, not the boolean thing',
            'test_cases': [
                {'function': 'is_even', 'input': 4, 'expected': 1},
                {'function': 'is_even', 'input': 7, 'expected': 0}
            ]
        },
        {
            'prompt': 'hmm but negatives are weird, fix that so it works properly',
            'test_cases': [
                {'function': 'is_even', 'input': -4, 'expected': 1},
                {'function': 'is_even', 'input': -3, 'expected': 0}
            ]
        },
        {
            'prompt': 'no go back to true/false but keep it working for negatives',
            'test_cases': [
                {'function': 'is_even', 'input': -4, 'expected': True},
                {'function': 'is_even', 'input': 7, 'expected': False}
            ]
        },
        
        # PROBLEM 2: String reverser
        {
            'prompt': 'i need a function that reverses a string',
            'test_cases': [
                {'function': 'reverse_string', 'input': 'hello', 'expected': 'olleh'}
            ]
        },
        {
            'prompt': 'no thats not right, keep first letter where it is',
            'test_cases': [
                {'function': 'reverse_string', 'input': 'hello', 'expected': 'holle'},
                {'function': 'reverse_string', 'input': 'test', 'expected': 'tset'}
            ]
        },
        {
            'prompt': 'wait make it uppercase too',
            'test_cases': [
                {'function': 'reverse_string', 'input': 'hello', 'expected': 'HOLLE'}
            ]
        },
        {
            'prompt': 'no just the vowels uppercase, rest lowercase',
            'test_cases': [
                {'function': 'reverse_string', 'input': 'Hello', 'expected': 'hOllE'},
                {'function': 'reverse_string', 'input': 'TEST', 'expected': 'tsEt'}
            ]
        },
        {
            'prompt': 'actually forget the first letter thing, just do the vowel uppercase thing but reversed',
            'test_cases': [
                {'function': 'reverse_string', 'input': 'hello', 'expected': 'OllEh'}
            ]
        },
        
        # PROBLEM 3: List sum
        {
            'prompt': 'sum all numbers in a list',
            'test_cases': [
                {'function': 'list_sum', 'input': [[1, 2, 3]], 'expected': 6}
            ]
        },
        {
            'prompt': 'what if theres negatives, ignore those',
            'test_cases': [
                {'function': 'list_sum', 'input': [[1, -2, 3]], 'expected': 4},
                {'function': 'list_sum', 'input': [[5, 6]], 'expected': 11}
            ]
        },
        {
            'prompt': 'no wait i meant skip numbers divisible by 3, not negatives',
            'test_cases': [
                {'function': 'list_sum', 'input': [[1, 3, 5]], 'expected': 6},
                {'function': 'list_sum', 'input': [[2, 4, 6]], 'expected': 6}
            ]
        },
        {
            'prompt': 'can you make it only sum the even positions, like 0, 2, 4',
            'test_cases': [
                {'function': 'list_sum', 'input': [[1, 2, 3, 4, 5]], 'expected': 9},
                {'function': 'list_sum', 'input': [[10, 1, 5]], 'expected': 15}
            ]
        },
        {
            'prompt': 'combine both - even positions AND skip multiples of 3',
            'test_cases': [
                {'function': 'list_sum', 'input': [[1, 2, 3, 4, 5, 6, 7]], 'expected': 13},
                {'function': 'list_sum', 'input': [[6, 1, 9, 2]], 'expected': 0}
            ]
        },
        
        # PROBLEM 4: Find max
        {
            'prompt': 'get the biggest number from a list',
            'test_cases': [
                {'function': 'find_max', 'input': [[1, 5, 3]], 'expected': 5}
            ]
        },
        {
            'prompt': 'what about if i want the second biggest not the biggest',
            'test_cases': [
                {'function': 'find_max', 'input': [[1, 5, 3]], 'expected': 3},
                {'function': 'find_max', 'input': [[10, 2, 8]], 'expected': 8}
            ]
        },
        {
            'prompt': 'no back to biggest but dont count single digit numbers',
            'test_cases': [
                {'function': 'find_max', 'input': [[5, 15, 3]], 'expected': 15},
                {'function': 'find_max', 'input': [[20, 8, 12]], 'expected': 20}
            ]
        },
        {
            'prompt': 'hmm what if theyre all single digits, return the biggest of those then',
            'test_cases': [
                {'function': 'find_max', 'input': [[1, 5, 3]], 'expected': 5},
                {'function': 'find_max', 'input': [[20, 8]], 'expected': 20}
            ]
        },
        {
            'prompt': 'actually just make it work for odd positions only, forget that other stuff',
            'test_cases': [
                {'function': 'find_max', 'input': [[10, 5, 20, 3]], 'expected': 5},
                {'function': 'find_max', 'input': [[1, 7, 2, 9]], 'expected': 9}
            ]
        },
        
        # PROBLEM 5: Factorial
        {
            'prompt': 'calculate factorial of a number',
            'test_cases': [
                {'function': 'factorial', 'input': 5, 'expected': 120},
                {'function': 'factorial', 'input': 3, 'expected': 6}
            ]
        },
        {
            'prompt': 'it should work for 0 too, thats 1',
            'test_cases': [
                {'function': 'factorial', 'input': 0, 'expected': 1},
                {'function': 'factorial', 'input': 4, 'expected': 24}
            ]
        },
        {
            'prompt': 'what about negative numbers, return -1 for those',
            'test_cases': [
                {'function': 'factorial', 'input': -5, 'expected': -1},
                {'function': 'factorial', 'input': 3, 'expected': 6}
            ]
        },
        {
            'prompt': 'can you skip the even numbers when multiplying',
            'test_cases': [
                {'function': 'factorial', 'input': 5, 'expected': 15},
                {'function': 'factorial', 'input': 4, 'expected': 3}
            ]
        },
        {
            'prompt': 'no forget that, normal factorial but if result over 100 just return 100',
            'test_cases': [
                {'function': 'factorial', 'input': 6, 'expected': 100},
                {'function': 'factorial', 'input': 4, 'expected': 24}
            ]
        },
        
        # PROBLEM 6: Vowel counter
        {
            'prompt': 'count how many vowels are in a string',
            'test_cases': [
                {'function': 'count_vowels', 'input': 'hello', 'expected': 2}
            ]
        },
        {
            'prompt': 'make it work for capitals too',
            'test_cases': [
                {'function': 'count_vowels', 'input': 'HELLO', 'expected': 2},
                {'function': 'count_vowels', 'input': 'HeLLo', 'expected': 2}
            ]
        },
        {
            'prompt': 'wait no, only count lowercase vowels, skip uppercase',
            'test_cases': [
                {'function': 'count_vowels', 'input': 'HeLLo', 'expected': 1},
                {'function': 'count_vowels', 'input': 'HELLO', 'expected': 0}
            ]
        },
        {
            'prompt': 'also count e twice for some reason',
            'test_cases': [
                {'function': 'count_vowels', 'input': 'hello', 'expected': 3},
                {'function': 'count_vowels', 'input': 'test', 'expected': 2}
            ]
        },
        {
            'prompt': 'no go back, all vowels normal but max is 3, dont go above 3',
            'test_cases': [
                {'function': 'count_vowels', 'input': 'aeiou', 'expected': 3},
                {'function': 'count_vowels', 'input': 'hello', 'expected': 2}
            ]
        },
        
        # PROBLEM 7: Palindrome
        {
            'prompt': 'check if a string is a palindrome',
            'test_cases': [
                {'function': 'is_palindrome', 'input': 'racecar', 'expected': True},
                {'function': 'is_palindrome', 'input': 'hello', 'expected': False}
            ]
        },
        {
            'prompt': 'spaces should be ignored',
            'test_cases': [
                {'function': 'is_palindrome', 'input': 'race car', 'expected': True},
                {'function': 'is_palindrome', 'input': 'hello', 'expected': False}
            ]
        },
        {
            'prompt': 'and capitals shouldnt matter either',
            'test_cases': [
                {'function': 'is_palindrome', 'input': 'RaceCar', 'expected': True},
                {'function': 'is_palindrome', 'input': 'Hello', 'expected': False}
            ]
        },
        {
            'prompt': 'actually only return true if its palindrome AND even length',
            'test_cases': [
                {'function': 'is_palindrome', 'input': 'abba', 'expected': True},
                {'function': 'is_palindrome', 'input': 'racecar', 'expected': False}
            ]
        },
        {
            'prompt': 'no forget length, but ignore first and last characters when checking',
            'test_cases': [
                {'function': 'is_palindrome', 'input': 'xracecary', 'expected': True},
                {'function': 'is_palindrome', 'input': 'hello', 'expected': False}
            ]
        },
        
        # PROBLEM 8: Remove duplicates
        {
            'prompt': 'remove duplicate numbers from a list',
            'test_cases': [
                {'function': 'remove_duplicates', 'input': [[1, 2, 2, 3]], 'expected': [1, 2, 3]}
            ]
        },
        {
            'prompt': 'can you sort it too',
            'test_cases': [
                {'function': 'remove_duplicates', 'input': [[3, 1, 2, 2]], 'expected': [1, 2, 3]}
            ]
        },
        {
            'prompt': 'wait keep only numbers that appear exactly twice',
            'test_cases': [
                {'function': 'remove_duplicates', 'input': [[1, 2, 2, 3, 3, 3]], 'expected': [2, 2]},
                {'function': 'remove_duplicates', 'input': [[5, 5, 4]], 'expected': [5, 5]}
            ]
        },
        {
            'prompt': 'no back to removing duplicates but also filter out anything under 5',
            'test_cases': [
                {'function': 'remove_duplicates', 'input': [[1, 7, 7, 3, 8]], 'expected': [7, 8]},
                {'function': 'remove_duplicates', 'input': [[10, 10, 2]], 'expected': [10]}
            ]
        },
        {
            'prompt': 'ok normal duplicate removal but reverse the order of the result',
            'test_cases': [
                {'function': 'remove_duplicates', 'input': [[1, 2, 2, 3]], 'expected': [3, 2, 1]}
            ]
        },
        
        # PROBLEM 9: Temperature
        {
            'prompt': 'convert celsius to fahrenheit',
            'test_cases': [
                {'function': 'celsius_to_fahrenheit', 'input': 0, 'expected': 32.0},
                {'function': 'celsius_to_fahrenheit', 'input': 100, 'expected': 212.0}
            ]
        },
        {
            'prompt': 'round it to whole numbers',
            'test_cases': [
                {'function': 'celsius_to_fahrenheit', 'input': 25, 'expected': 77},
                {'function': 'celsius_to_fahrenheit', 'input': 30, 'expected': 86}
            ]
        },
        {
            'prompt': 'if its negative input add 5 to the result',
            'test_cases': [
                {'function': 'celsius_to_fahrenheit', 'input': -10, 'expected': 19},
                {'function': 'celsius_to_fahrenheit', 'input': 0, 'expected': 32}
            ]
        },
        {
            'prompt': 'wait no, cap it at 100 max, dont go above that',
            'test_cases': [
                {'function': 'celsius_to_fahrenheit', 'input': 50, 'expected': 100},
                {'function': 'celsius_to_fahrenheit', 'input': 20, 'expected': 68}
            ]
        },
        {
            'prompt': 'actually remove the cap but subtract 10 if input is even',
            'test_cases': [
                {'function': 'celsius_to_fahrenheit', 'input': 0, 'expected': 22},
                {'function': 'celsius_to_fahrenheit', 'input': 100, 'expected': 202},
                {'function': 'celsius_to_fahrenheit', 'input': 25, 'expected': 77}
            ]
        },
        
        # PROBLEM 10: Prime checker
        {
            'prompt': 'check if number is prime',
            'test_cases': [
                {'function': 'is_prime', 'input': 7, 'expected': True},
                {'function': 'is_prime', 'input': 4, 'expected': False}
            ]
        },
        {
            'prompt': 'what about 2, thats prime right',
            'test_cases': [
                {'function': 'is_prime', 'input': 2, 'expected': True},
                {'function': 'is_prime', 'input': 7, 'expected': True}
            ]
        },
        {
            'prompt': 'and negatives should return false',
            'test_cases': [
                {'function': 'is_prime', 'input': -7, 'expected': False},
                {'function': 'is_prime', 'input': 7, 'expected': True}
            ]
        },
        {
            'prompt': 'actually only return true if prime AND bigger than 10',
            'test_cases': [
                {'function': 'is_prime', 'input': 11, 'expected': True},
                {'function': 'is_prime', 'input': 7, 'expected': False}
            ]
        },
        {
            'prompt': 'no forget that, normal prime check but false if the digits add up to an even number',
            'test_cases': [
                {'function': 'is_prime', 'input': 13, 'expected': False},
                {'function': 'is_prime', 'input': 11, 'expected': True},
                {'function': 'is_prime', 'input': 4, 'expected': False}
            ]
        }
    ]
    
    return jsonify({'prompts': prompts})

if __name__ == '__main__':
    app.run(port=8080, debug=True)
