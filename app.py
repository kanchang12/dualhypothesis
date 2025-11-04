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
    
    attempts = []
    passed = False
    max_attempts = 10
    
    generation_prompt = f"""Generate a Python function that EXACTLY matches this requirement.

Requirement: {prompt_text}

CRITICAL RULES:
1. Function name must be EXACTLY as specified
2. Function must handle ALL edge cases
3. Return type must match expected output
4. No print statements, no extra functions, no comments
5. Just the function code

Generate ONLY the function code:"""
    
    for attempt in range(max_attempts):
        start = time.time()
        
        # Generate with Gemini
        response = gemini.generate_content(generation_prompt)
        code = response.text.strip()
        
        # Clean markdown if present
        if '```python' in code:
            code = code.split('```python')[1].split('```')[0].strip()
        elif '```' in code:
            code = code.split('```')[1].split('```')[0].strip()
        
        elapsed = time.time() - start
        
        # Calculate costs
        in_tokens = count_tokens(generation_prompt)
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
            break
    
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
        'details': attempts
    })

@app.route('/api/path2', methods=['POST'])
def path2():
    """Path 2: Gemini → OpenAI Validation → Unit Test → Retry if fail"""
    data = request.json
    prompt_text = data['prompt']
    test_cases = data['test_cases']
    
    attempts = []
    passed = False
    max_attempts = 10
    
    generation_prompt = f"""Generate a Python function that EXACTLY matches this requirement.

Requirement: {prompt_text}

CRITICAL RULES:
1. Function name must be EXACTLY as specified
2. Function must handle ALL edge cases
3. Return type must match expected output
4. No print statements, no extra functions, no comments
5. Just the function code

Generate ONLY the function code:"""
    
    for attempt in range(max_attempts):
        # Step 1: Generate with Gemini
        gemini_start = time.time()
        response = gemini.generate_content(generation_prompt)
        code = response.text.strip()
        
        # Clean markdown
        if '```python' in code:
            code = code.split('```python')[1].split('```')[0].strip()
        elif '```' in code:
            code = code.split('```')[1].split('```')[0].strip()
        
        gemini_time = time.time() - gemini_start
        gemini_in = count_tokens(generation_prompt)
        gemini_out = count_tokens(code)
        gemini_cost = (gemini_in * GEMINI_INPUT_COST) + (gemini_out * GEMINI_OUTPUT_COST)
        
        # Step 2: OpenAI validates
        validation_prompt = f"""You are a code validator. Check if this code meets the requirement.

REQUIREMENT: {prompt_text}

CODE:
{code}

Check:
1. Does it fulfill the requirement?
2. Is it complete?
3. No obvious errors?

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
            break
    
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
        'details': attempts
    })

@app.route('/api/prompts', methods=['GET'])
def get_prompts():
    """Return 50 prompts: 10 base problems with 5 modifications each"""
    prompts = [
        # PROBLEM 1: Even checker - 5 variations
        {
            'prompt': 'Write a Python function called is_even that checks if a number is even',
            'test_cases': [
                {'function': 'is_even', 'input': 4, 'expected': True},
                {'function': 'is_even', 'input': 7, 'expected': False}
            ]
        },
        {
            'prompt': 'Write a Python function called is_even that checks if a number is even. Change the parameter name to "num"',
            'test_cases': [
                {'function': 'is_even', 'input': 4, 'expected': True},
                {'function': 'is_even', 'input': 7, 'expected': False}
            ]
        },
        {
            'prompt': 'Write a Python function called is_even that checks if a number is even. Add support for negative numbers',
            'test_cases': [
                {'function': 'is_even', 'input': -4, 'expected': True},
                {'function': 'is_even', 'input': -7, 'expected': False}
            ]
        },
        {
            'prompt': 'Write a Python function called is_even that checks if a number is even. Change return True to return "Yes" and False to "No"',
            'test_cases': [
                {'function': 'is_even', 'input': 4, 'expected': 'Yes'},
                {'function': 'is_even', 'input': 7, 'expected': 'No'}
            ]
        },
        {
            'prompt': 'Write a Python function called is_even that checks if a number is even. Add a check: return None if input is 0',
            'test_cases': [
                {'function': 'is_even', 'input': 0, 'expected': None},
                {'function': 'is_even', 'input': 4, 'expected': True}
            ]
        },
        
        # PROBLEM 2: String reverser - 5 variations
        {
            'prompt': 'Write a Python function called reverse_string that reverses a string',
            'test_cases': [
                {'function': 'reverse_string', 'input': 'hello', 'expected': 'olleh'}
            ]
        },
        {
            'prompt': 'Write a Python function called reverse_string that reverses a string. Change parameter name to "text"',
            'test_cases': [
                {'function': 'reverse_string', 'input': 'hello', 'expected': 'olleh'}
            ]
        },
        {
            'prompt': 'Write a Python function called reverse_string that reverses a string. Make it return uppercase',
            'test_cases': [
                {'function': 'reverse_string', 'input': 'hello', 'expected': 'OLLEH'}
            ]
        },
        {
            'prompt': 'Write a Python function called reverse_string that reverses a string. Add: if empty string, return "EMPTY"',
            'test_cases': [
                {'function': 'reverse_string', 'input': '', 'expected': 'EMPTY'},
                {'function': 'reverse_string', 'input': 'hi', 'expected': 'ih'}
            ]
        },
        {
            'prompt': 'Write a Python function called reverse_string that reverses a string. Change: only reverse if length > 3, else return as-is',
            'test_cases': [
                {'function': 'reverse_string', 'input': 'hi', 'expected': 'hi'},
                {'function': 'reverse_string', 'input': 'hello', 'expected': 'olleh'}
            ]
        },
        
        # PROBLEM 3: List sum - 5 variations
        {
            'prompt': 'Write a Python function called list_sum that returns sum of a list',
            'test_cases': [
                {'function': 'list_sum', 'input': [[1, 2, 3]], 'expected': 6}
            ]
        },
        {
            'prompt': 'Write a Python function called list_sum that returns sum of a list. Change parameter to "numbers"',
            'test_cases': [
                {'function': 'list_sum', 'input': [[1, 2, 3]], 'expected': 6}
            ]
        },
        {
            'prompt': 'Write a Python function called list_sum that returns sum of a list. Return 0 if list is empty',
            'test_cases': [
                {'function': 'list_sum', 'input': [[]], 'expected': 0},
                {'function': 'list_sum', 'input': [[5]], 'expected': 5}
            ]
        },
        {
            'prompt': 'Write a Python function called list_sum that returns sum of a list. Multiply the sum by 2 before returning',
            'test_cases': [
                {'function': 'list_sum', 'input': [[1, 2, 3]], 'expected': 12}
            ]
        },
        {
            'prompt': 'Write a Python function called list_sum that returns sum of a list. Only sum positive numbers, ignore negatives',
            'test_cases': [
                {'function': 'list_sum', 'input': [[1, -2, 3]], 'expected': 4},
                {'function': 'list_sum', 'input': [[-1, -2]], 'expected': 0}
            ]
        },
        
        # PROBLEM 4: Factorial - 5 variations
        {
            'prompt': 'Write a Python function called factorial that calculates factorial',
            'test_cases': [
                {'function': 'factorial', 'input': 5, 'expected': 120},
                {'function': 'factorial', 'input': 0, 'expected': 1}
            ]
        },
        {
            'prompt': 'Write a Python function called factorial that calculates factorial. Change parameter to "n"',
            'test_cases': [
                {'function': 'factorial', 'input': 5, 'expected': 120}
            ]
        },
        {
            'prompt': 'Write a Python function called factorial that calculates factorial. Return -1 for negative inputs',
            'test_cases': [
                {'function': 'factorial', 'input': -3, 'expected': -1},
                {'function': 'factorial', 'input': 3, 'expected': 6}
            ]
        },
        {
            'prompt': 'Write a Python function called factorial that calculates factorial. Add 10 to the result before returning',
            'test_cases': [
                {'function': 'factorial', 'input': 3, 'expected': 16},
                {'function': 'factorial', 'input': 4, 'expected': 34}
            ]
        },
        {
            'prompt': 'Write a Python function called factorial that calculates factorial. Return None if input is greater than 10',
            'test_cases': [
                {'function': 'factorial', 'input': 11, 'expected': None},
                {'function': 'factorial', 'input': 5, 'expected': 120}
            ]
        },
        
        # PROBLEM 5: Max finder - 5 variations
        {
            'prompt': 'Write a Python function called find_max that returns maximum from a list',
            'test_cases': [
                {'function': 'find_max', 'input': [[1, 5, 3]], 'expected': 5}
            ]
        },
        {
            'prompt': 'Write a Python function called find_max that returns maximum from a list. Change parameter to "nums"',
            'test_cases': [
                {'function': 'find_max', 'input': [[1, 5, 3]], 'expected': 5}
            ]
        },
        {
            'prompt': 'Write a Python function called find_max that returns maximum from a list. Return None if list is empty',
            'test_cases': [
                {'function': 'find_max', 'input': [[]], 'expected': None},
                {'function': 'find_max', 'input': [[7]], 'expected': 7}
            ]
        },
        {
            'prompt': 'Write a Python function called find_max that returns maximum from a list. Return the maximum minus 1',
            'test_cases': [
                {'function': 'find_max', 'input': [[1, 5, 3]], 'expected': 4}
            ]
        },
        {
            'prompt': 'Write a Python function called find_max that returns maximum from a list. Ignore negative numbers, only find max among positives',
            'test_cases': [
                {'function': 'find_max', 'input': [[1, -5, 3]], 'expected': 3},
                {'function': 'find_max', 'input': [[-1, -2]], 'expected': None}
            ]
        },
        
        # PROBLEM 6: Vowel counter - 5 variations
        {
            'prompt': 'Write a Python function called count_vowels that counts vowels in a string',
            'test_cases': [
                {'function': 'count_vowels', 'input': 'hello', 'expected': 2}
            ]
        },
        {
            'prompt': 'Write a Python function called count_vowels that counts vowels in a string. Change parameter to "text"',
            'test_cases': [
                {'function': 'count_vowels', 'input': 'hello', 'expected': 2}
            ]
        },
        {
            'prompt': 'Write a Python function called count_vowels that counts vowels in a string. Make it case-insensitive (count A and a)',
            'test_cases': [
                {'function': 'count_vowels', 'input': 'HELLO', 'expected': 2},
                {'function': 'count_vowels', 'input': 'AEIOUaeiou', 'expected': 10}
            ]
        },
        {
            'prompt': 'Write a Python function called count_vowels that counts vowels in a string. Multiply the count by 2',
            'test_cases': [
                {'function': 'count_vowels', 'input': 'hello', 'expected': 4}
            ]
        },
        {
            'prompt': 'Write a Python function called count_vowels that counts vowels in a string. Return 0 if string has no vowels',
            'test_cases': [
                {'function': 'count_vowels', 'input': 'xyz', 'expected': 0},
                {'function': 'count_vowels', 'input': 'aaa', 'expected': 3}
            ]
        },
        
        # PROBLEM 7: Palindrome checker - 5 variations
        {
            'prompt': 'Write a Python function called is_palindrome that checks if string is palindrome',
            'test_cases': [
                {'function': 'is_palindrome', 'input': 'racecar', 'expected': True},
                {'function': 'is_palindrome', 'input': 'hello', 'expected': False}
            ]
        },
        {
            'prompt': 'Write a Python function called is_palindrome that checks if string is palindrome. Change parameter to "word"',
            'test_cases': [
                {'function': 'is_palindrome', 'input': 'racecar', 'expected': True}
            ]
        },
        {
            'prompt': 'Write a Python function called is_palindrome that checks if string is palindrome. Make it case-insensitive',
            'test_cases': [
                {'function': 'is_palindrome', 'input': 'Racecar', 'expected': True},
                {'function': 'is_palindrome', 'input': 'RaceCar', 'expected': True}
            ]
        },
        {
            'prompt': 'Write a Python function called is_palindrome that checks if string is palindrome. Return "YES" or "NO" instead of True/False',
            'test_cases': [
                {'function': 'is_palindrome', 'input': 'racecar', 'expected': 'YES'},
                {'function': 'is_palindrome', 'input': 'hello', 'expected': 'NO'}
            ]
        },
        {
            'prompt': 'Write a Python function called is_palindrome that checks if string is palindrome. Return None if string length is less than 2',
            'test_cases': [
                {'function': 'is_palindrome', 'input': 'a', 'expected': None},
                {'function': 'is_palindrome', 'input': 'aa', 'expected': True}
            ]
        },
        
        # PROBLEM 8: List duplicates remover - 5 variations
        {
            'prompt': 'Write a Python function called remove_duplicates that removes duplicates from list',
            'test_cases': [
                {'function': 'remove_duplicates', 'input': [[1, 2, 2, 3]], 'expected': [1, 2, 3]}
            ]
        },
        {
            'prompt': 'Write a Python function called remove_duplicates that removes duplicates from list. Change parameter to "items"',
            'test_cases': [
                {'function': 'remove_duplicates', 'input': [[1, 2, 2, 3]], 'expected': [1, 2, 3]}
            ]
        },
        {
            'prompt': 'Write a Python function called remove_duplicates that removes duplicates from list. Sort the result',
            'test_cases': [
                {'function': 'remove_duplicates', 'input': [[3, 1, 2, 2]], 'expected': [1, 2, 3]}
            ]
        },
        {
            'prompt': 'Write a Python function called remove_duplicates that removes duplicates from list. Return empty list if input is empty',
            'test_cases': [
                {'function': 'remove_duplicates', 'input': [[]], 'expected': []},
                {'function': 'remove_duplicates', 'input': [[1, 1]], 'expected': [1]}
            ]
        },
        {
            'prompt': 'Write a Python function called remove_duplicates that removes duplicates from list. Keep only numbers greater than 0',
            'test_cases': [
                {'function': 'remove_duplicates', 'input': [[-1, 2, 2, 3]], 'expected': [2, 3]},
                {'function': 'remove_duplicates', 'input': [[0, 1, 1]], 'expected': [1]}
            ]
        },
        
        # PROBLEM 9: Temperature converter - 5 variations
        {
            'prompt': 'Write a Python function called celsius_to_fahrenheit that converts Celsius to Fahrenheit',
            'test_cases': [
                {'function': 'celsius_to_fahrenheit', 'input': 0, 'expected': 32.0},
                {'function': 'celsius_to_fahrenheit', 'input': 100, 'expected': 212.0}
            ]
        },
        {
            'prompt': 'Write a Python function called celsius_to_fahrenheit that converts Celsius to Fahrenheit. Change parameter to "temp"',
            'test_cases': [
                {'function': 'celsius_to_fahrenheit', 'input': 0, 'expected': 32.0}
            ]
        },
        {
            'prompt': 'Write a Python function called celsius_to_fahrenheit that converts Celsius to Fahrenheit. Round result to 1 decimal place',
            'test_cases': [
                {'function': 'celsius_to_fahrenheit', 'input': 25, 'expected': 77.0}
            ]
        },
        {
            'prompt': 'Write a Python function called celsius_to_fahrenheit that converts Celsius to Fahrenheit. Return None if input is below -273 (absolute zero)',
            'test_cases': [
                {'function': 'celsius_to_fahrenheit', 'input': -274, 'expected': None},
                {'function': 'celsius_to_fahrenheit', 'input': 0, 'expected': 32.0}
            ]
        },
        {
            'prompt': 'Write a Python function called celsius_to_fahrenheit that converts Celsius to Fahrenheit. Add 10 to the result',
            'test_cases': [
                {'function': 'celsius_to_fahrenheit', 'input': 0, 'expected': 42.0}
            ]
        },
        
        # PROBLEM 10: Prime checker - 5 variations
        {
            'prompt': 'Write a Python function called is_prime that checks if number is prime',
            'test_cases': [
                {'function': 'is_prime', 'input': 7, 'expected': True},
                {'function': 'is_prime', 'input': 4, 'expected': False}
            ]
        },
        {
            'prompt': 'Write a Python function called is_prime that checks if number is prime. Change parameter to "num"',
            'test_cases': [
                {'function': 'is_prime', 'input': 7, 'expected': True}
            ]
        },
        {
            'prompt': 'Write a Python function called is_prime that checks if number is prime. Return False for numbers less than 2',
            'test_cases': [
                {'function': 'is_prime', 'input': 1, 'expected': False},
                {'function': 'is_prime', 'input': 0, 'expected': False},
                {'function': 'is_prime', 'input': 2, 'expected': True}
            ]
        },
        {
            'prompt': 'Write a Python function called is_prime that checks if number is prime. Return "PRIME" or "NOT PRIME" instead of True/False',
            'test_cases': [
                {'function': 'is_prime', 'input': 7, 'expected': 'PRIME'},
                {'function': 'is_prime', 'input': 4, 'expected': 'NOT PRIME'}
            ]
        },
        {
            'prompt': 'Write a Python function called is_prime that checks if number is prime. Return None if input is negative',
            'test_cases': [
                {'function': 'is_prime', 'input': -5, 'expected': None},
                {'function': 'is_prime', 'input': 5, 'expected': True}
            ]
        }
    ]
    
    return jsonify({'prompts': prompts})

if __name__ == '__main__':
    app.run(port=8080, debug=True)
