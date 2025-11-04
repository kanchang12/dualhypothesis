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
    """Return 50 prompts: 10 base problems with 5 HARD modifications each"""
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
            'prompt': 'Write a Python function called is_even that checks if a number is even. But return False for 0 (zero is special case)',
            'test_cases': [
                {'function': 'is_even', 'input': 0, 'expected': False},
                {'function': 'is_even', 'input': 4, 'expected': True},
                {'function': 'is_even', 'input': 7, 'expected': False}
            ]
        },
        {
            'prompt': 'Write a Python function called is_even. Return True only if number is even AND positive. Negative evens return False',
            'test_cases': [
                {'function': 'is_even', 'input': 4, 'expected': True},
                {'function': 'is_even', 'input': -4, 'expected': False},
                {'function': 'is_even', 'input': 7, 'expected': False}
            ]
        },
        {
            'prompt': 'Write a Python function called is_even. Return 1 if even, 0 if odd, but return -1 if number is divisible by 4',
            'test_cases': [
                {'function': 'is_even', 'input': 8, 'expected': -1},
                {'function': 'is_even', 'input': 6, 'expected': 1},
                {'function': 'is_even', 'input': 7, 'expected': 0}
            ]
        },
        {
            'prompt': 'Write a Python function called is_even. Return True for even numbers except multiples of 10 (return False for those)',
            'test_cases': [
                {'function': 'is_even', 'input': 20, 'expected': False},
                {'function': 'is_even', 'input': 8, 'expected': True},
                {'function': 'is_even', 'input': 7, 'expected': False}
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
            'prompt': 'Write a Python function called reverse_string that reverses a string. But keep the first character in place, only reverse the rest',
            'test_cases': [
                {'function': 'reverse_string', 'input': 'hello', 'expected': 'holle'},
                {'function': 'reverse_string', 'input': 'abc', 'expected': 'acb'}
            ]
        },
        {
            'prompt': 'Write a Python function called reverse_string. Reverse it but convert vowels to uppercase and consonants to lowercase',
            'test_cases': [
                {'function': 'reverse_string', 'input': 'Hello', 'expected': 'OllEh'},
                {'function': 'reverse_string', 'input': 'test', 'expected': 'tsEt'}
            ]
        },
        {
            'prompt': 'Write a Python function called reverse_string. Reverse every other character only (positions 1,3,5...), keep rest same',
            'test_cases': [
                {'function': 'reverse_string', 'input': 'abcd', 'expected': 'adcb'},
                {'function': 'reverse_string', 'input': 'hello', 'expected': 'holle'}
            ]
        },
        {
            'prompt': 'Write a Python function called reverse_string. Reverse but skip any digits, keep them in original position',
            'test_cases': [
                {'function': 'reverse_string', 'input': 'a1b2c', 'expected': 'c1b2a'},
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
            'prompt': 'Write a Python function called list_sum. Sum the list but skip any number that is a multiple of 3',
            'test_cases': [
                {'function': 'list_sum', 'input': [[1, 2, 3, 4]], 'expected': 7},
                {'function': 'list_sum', 'input': [[6, 1, 9]], 'expected': 1}
            ]
        },
        {
            'prompt': 'Write a Python function called list_sum. Sum only numbers at even indices (0,2,4...), ignore odd indices',
            'test_cases': [
                {'function': 'list_sum', 'input': [[1, 2, 3, 4, 5]], 'expected': 9},
                {'function': 'list_sum', 'input': [[10, 1, 5]], 'expected': 15}
            ]
        },
        {
            'prompt': 'Write a Python function called list_sum. Sum the list but if sum exceeds 10, return 10 instead',
            'test_cases': [
                {'function': 'list_sum', 'input': [[5, 8]], 'expected': 10},
                {'function': 'list_sum', 'input': [[2, 3]], 'expected': 5}
            ]
        },
        {
            'prompt': 'Write a Python function called list_sum. Sum numbers but subtract the smallest number from the total',
            'test_cases': [
                {'function': 'list_sum', 'input': [[5, 2, 8]], 'expected': 13},
                {'function': 'list_sum', 'input': [[10, 1, 4]], 'expected': 14}
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
            'prompt': 'Write a Python function called factorial. Calculate factorial but multiply result by the input number one extra time',
            'test_cases': [
                {'function': 'factorial', 'input': 4, 'expected': 96},
                {'function': 'factorial', 'input': 3, 'expected': 18}
            ]
        },
        {
            'prompt': 'Write a Python function called factorial. Calculate factorial but skip multiplying by even numbers in the sequence',
            'test_cases': [
                {'function': 'factorial', 'input': 5, 'expected': 15},
                {'function': 'factorial', 'input': 4, 'expected': 3}
            ]
        },
        {
            'prompt': 'Write a Python function called factorial. Return factorial but if result is greater than 100, return 100 instead',
            'test_cases': [
                {'function': 'factorial', 'input': 5, 'expected': 100},
                {'function': 'factorial', 'input': 3, 'expected': 6}
            ]
        },
        {
            'prompt': 'Write a Python function called factorial. Calculate factorial but add the sum of all numbers from 1 to n to the result',
            'test_cases': [
                {'function': 'factorial', 'input': 4, 'expected': 34},
                {'function': 'factorial', 'input': 3, 'expected': 12}
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
            'prompt': 'Write a Python function called find_max. Return the maximum but ignore any single-digit numbers (0-9)',
            'test_cases': [
                {'function': 'find_max', 'input': [[1, 15, 3]], 'expected': 15},
                {'function': 'find_max', 'input': [[20, 5, 18]], 'expected': 20}
            ]
        },
        {
            'prompt': 'Write a Python function called find_max. Return second highest number, not the highest',
            'test_cases': [
                {'function': 'find_max', 'input': [[1, 5, 3]], 'expected': 3},
                {'function': 'find_max', 'input': [[10, 2, 8]], 'expected': 8}
            ]
        },
        {
            'prompt': 'Write a Python function called find_max. Return max but only among numbers at odd indices (1,3,5...)',
            'test_cases': [
                {'function': 'find_max', 'input': [[10, 2, 8, 9]], 'expected': 9},
                {'function': 'find_max', 'input': [[1, 5, 3, 4]], 'expected': 5}
            ]
        },
        {
            'prompt': 'Write a Python function called find_max. Return the maximum but subtract 5 if it is greater than 10',
            'test_cases': [
                {'function': 'find_max', 'input': [[1, 15, 3]], 'expected': 10},
                {'function': 'find_max', 'input': [[1, 8, 3]], 'expected': 8}
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
            'prompt': 'Write a Python function called count_vowels. Count vowels but only lowercase ones, skip uppercase vowels',
            'test_cases': [
                {'function': 'count_vowels', 'input': 'hEllo', 'expected': 1},
                {'function': 'count_vowels', 'input': 'HELLO', 'expected': 0}
            ]
        },
        {
            'prompt': 'Write a Python function called count_vowels. Count vowels but double-count any letter "e"',
            'test_cases': [
                {'function': 'count_vowels', 'input': 'hello', 'expected': 3},
                {'function': 'count_vowels', 'input': 'test', 'expected': 2}
            ]
        },
        {
            'prompt': 'Write a Python function called count_vowels. Count vowels but only at even positions (0,2,4...)',
            'test_cases': [
                {'function': 'count_vowels', 'input': 'hello', 'expected': 1},
                {'function': 'count_vowels', 'input': 'aeiou', 'expected': 3}
            ]
        },
        {
            'prompt': 'Write a Python function called count_vowels. Count vowels but if there are more than 3, return 3 (max 3)',
            'test_cases': [
                {'function': 'count_vowels', 'input': 'aeiou', 'expected': 3},
                {'function': 'count_vowels', 'input': 'hello', 'expected': 2}
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
            'prompt': 'Write a Python function called is_palindrome. Check palindrome but ignore spaces and punctuation',
            'test_cases': [
                {'function': 'is_palindrome', 'input': 'race car', 'expected': True},
                {'function': 'is_palindrome', 'input': 'a man a', 'expected': True},
                {'function': 'is_palindrome', 'input': 'hello', 'expected': False}
            ]
        },
        {
            'prompt': 'Write a Python function called is_palindrome. Return True only if it is palindrome AND has even length',
            'test_cases': [
                {'function': 'is_palindrome', 'input': 'abba', 'expected': True},
                {'function': 'is_palindrome', 'input': 'racecar', 'expected': False},
                {'function': 'is_palindrome', 'input': 'hello', 'expected': False}
            ]
        },
        {
            'prompt': 'Write a Python function called is_palindrome. Check palindrome ignoring first and last characters',
            'test_cases': [
                {'function': 'is_palindrome', 'input': 'xracecary', 'expected': True},
                {'function': 'is_palindrome', 'input': 'hello', 'expected': False}
            ]
        },
        {
            'prompt': 'Write a Python function called is_palindrome. Return True if palindrome OR if length is 1, otherwise False',
            'test_cases': [
                {'function': 'is_palindrome', 'input': 'a', 'expected': True},
                {'function': 'is_palindrome', 'input': 'racecar', 'expected': True},
                {'function': 'is_palindrome', 'input': 'ab', 'expected': False}
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
            'prompt': 'Write a Python function called remove_duplicates. Remove duplicates but keep numbers that appear exactly twice',
            'test_cases': [
                {'function': 'remove_duplicates', 'input': [[1, 2, 2, 3, 3, 3]], 'expected': [1, 2, 2]},
                {'function': 'remove_duplicates', 'input': [[5, 5, 4]], 'expected': [5, 5]}
            ]
        },
        {
            'prompt': 'Write a Python function called remove_duplicates. Remove duplicates but only keep numbers greater than 5',
            'test_cases': [
                {'function': 'remove_duplicates', 'input': [[1, 7, 7, 3, 8]], 'expected': [7, 8]},
                {'function': 'remove_duplicates', 'input': [[10, 10, 2]], 'expected': [10]}
            ]
        },
        {
            'prompt': 'Write a Python function called remove_duplicates. Remove duplicates and return in reverse order',
            'test_cases': [
                {'function': 'remove_duplicates', 'input': [[1, 2, 2, 3]], 'expected': [3, 2, 1]},
                {'function': 'remove_duplicates', 'input': [[5, 1, 5]], 'expected': [1, 5]}
            ]
        },
        {
            'prompt': 'Write a Python function called remove_duplicates. Remove duplicates but multiply each unique number by 2',
            'test_cases': [
                {'function': 'remove_duplicates', 'input': [[1, 2, 2, 3]], 'expected': [2, 4, 6]},
                {'function': 'remove_duplicates', 'input': [[5, 5]], 'expected': [10]}
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
            'prompt': 'Write a Python function called celsius_to_fahrenheit. Convert but add 5 degrees if input is negative',
            'test_cases': [
                {'function': 'celsius_to_fahrenheit', 'input': -10, 'expected': 19.0},
                {'function': 'celsius_to_fahrenheit', 'input': 0, 'expected': 32.0}
            ]
        },
        {
            'prompt': 'Write a Python function called celsius_to_fahrenheit. Convert but return only integer part (no decimals)',
            'test_cases': [
                {'function': 'celsius_to_fahrenheit', 'input': 25, 'expected': 77},
                {'function': 'celsius_to_fahrenheit', 'input': 30, 'expected': 86}
            ]
        },
        {
            'prompt': 'Write a Python function called celsius_to_fahrenheit. Convert but if result is above 100F, return 100',
            'test_cases': [
                {'function': 'celsius_to_fahrenheit', 'input': 50, 'expected': 100},
                {'function': 'celsius_to_fahrenheit', 'input': 20, 'expected': 68.0}
            ]
        },
        {
            'prompt': 'Write a Python function called celsius_to_fahrenheit. Convert but subtract 10 from result if input is even number',
            'test_cases': [
                {'function': 'celsius_to_fahrenheit', 'input': 0, 'expected': 22.0},
                {'function': 'celsius_to_fahrenheit', 'input': 100, 'expected': 202.0},
                {'function': 'celsius_to_fahrenheit', 'input': 25, 'expected': 77.0}
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
            'prompt': 'Write a Python function called is_prime. Check if prime but return False if number is 2 (exception)',
            'test_cases': [
                {'function': 'is_prime', 'input': 2, 'expected': False},
                {'function': 'is_prime', 'input': 7, 'expected': True},
                {'function': 'is_prime', 'input': 4, 'expected': False}
            ]
        },
        {
            'prompt': 'Write a Python function called is_prime. Return True if prime AND greater than 10, else False',
            'test_cases': [
                {'function': 'is_prime', 'input': 11, 'expected': True},
                {'function': 'is_prime', 'input': 7, 'expected': False},
                {'function': 'is_prime', 'input': 4, 'expected': False}
            ]
        },
        {
            'prompt': 'Write a Python function called is_prime. Check if prime but also return True if number is perfect square',
            'test_cases': [
                {'function': 'is_prime', 'input': 9, 'expected': True},
                {'function': 'is_prime', 'input': 7, 'expected': True},
                {'function': 'is_prime', 'input': 6, 'expected': False}
            ]
        },
        {
            'prompt': 'Write a Python function called is_prime. Return True if prime, but False if sum of digits is even',
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
