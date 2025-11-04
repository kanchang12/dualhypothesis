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
    """Execute code and run test cases"""
    try:
        # Create isolated namespace
        namespace = {}
        
        # Execute the code
        exec(code, namespace)
        
        # Run each test case
        for test in test_cases:
            func_name = test['function']
            inputs = test['input']
            expected = test['expected']
            
            if func_name not in namespace:
                return False, f"Function {func_name} not found"
            
            # Call function with inputs
            if isinstance(inputs, list):
                result = namespace[func_name](*inputs)
            else:
                result = namespace[func_name](inputs)
            
            # Check result
            if result != expected:
                return False, f"Expected {expected}, got {result}"
        
        return True, "All tests passed"
        
    except Exception as e:
        return False, str(e)

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
    
    generation_prompt = f"""You are a code generator. Generate ONLY working Python code.

Requirement: {prompt_text}

Generate only the code, no explanations, no markdown."""
    
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
    
    generation_prompt = f"""You are a code generator. Generate ONLY working Python code.

Requirement: {prompt_text}

Generate only the code, no explanations, no markdown."""
    
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
    """Return 50 prompts with unit tests"""
    prompts = [
        # Simple math functions (10)
        {
            'prompt': 'Write a Python function called is_even that checks if a number is even',
            'test_cases': [
                {'function': 'is_even', 'input': 4, 'expected': True},
                {'function': 'is_even', 'input': 7, 'expected': False},
                {'function': 'is_even', 'input': 0, 'expected': True}
            ]
        },
        {
            'prompt': 'Write a Python function called add_numbers that adds two numbers',
            'test_cases': [
                {'function': 'add_numbers', 'input': [5, 3], 'expected': 8},
                {'function': 'add_numbers', 'input': [10, -5], 'expected': 5}
            ]
        },
        {
            'prompt': 'Write a Python function called multiply that multiplies two numbers',
            'test_cases': [
                {'function': 'multiply', 'input': [4, 5], 'expected': 20},
                {'function': 'multiply', 'input': [7, 0], 'expected': 0}
            ]
        },
        {
            'prompt': 'Write a Python function called max_of_three that finds maximum of three numbers',
            'test_cases': [
                {'function': 'max_of_three', 'input': [1, 5, 3], 'expected': 5},
                {'function': 'max_of_three', 'input': [10, 2, 8], 'expected': 10}
            ]
        },
        {
            'prompt': 'Write a Python function called absolute_value that returns absolute value',
            'test_cases': [
                {'function': 'absolute_value', 'input': -5, 'expected': 5},
                {'function': 'absolute_value', 'input': 3, 'expected': 3}
            ]
        },
        {
            'prompt': 'Write a Python function called is_positive that checks if number is positive',
            'test_cases': [
                {'function': 'is_positive', 'input': 5, 'expected': True},
                {'function': 'is_positive', 'input': -3, 'expected': False},
                {'function': 'is_positive', 'input': 0, 'expected': False}
            ]
        },
        {
            'prompt': 'Write a Python function called square that returns square of a number',
            'test_cases': [
                {'function': 'square', 'input': 4, 'expected': 16},
                {'function': 'square', 'input': 0, 'expected': 0}
            ]
        },
        {
            'prompt': 'Write a Python function called is_divisible that checks if first number is divisible by second',
            'test_cases': [
                {'function': 'is_divisible', 'input': [10, 2], 'expected': True},
                {'function': 'is_divisible', 'input': [7, 3], 'expected': False}
            ]
        },
        {
            'prompt': 'Write a Python function called celsius_to_fahrenheit that converts Celsius to Fahrenheit',
            'test_cases': [
                {'function': 'celsius_to_fahrenheit', 'input': 0, 'expected': 32.0},
                {'function': 'celsius_to_fahrenheit', 'input': 100, 'expected': 212.0}
            ]
        },
        {
            'prompt': 'Write a Python function called power that raises first number to power of second',
            'test_cases': [
                {'function': 'power', 'input': [2, 3], 'expected': 8},
                {'function': 'power', 'input': [5, 0], 'expected': 1}
            ]
        },
        
        # String functions (15)
        {
            'prompt': 'Write a Python function called reverse_string that reverses a string',
            'test_cases': [
                {'function': 'reverse_string', 'input': 'hello', 'expected': 'olleh'},
                {'function': 'reverse_string', 'input': 'test', 'expected': 'tset'}
            ]
        },
        {
            'prompt': 'Write a Python function called count_vowels that counts vowels in a string',
            'test_cases': [
                {'function': 'count_vowels', 'input': 'hello', 'expected': 2},
                {'function': 'count_vowels', 'input': 'programming', 'expected': 3}
            ]
        },
        {
            'prompt': 'Write a Python function called is_palindrome that checks if string is palindrome',
            'test_cases': [
                {'function': 'is_palindrome', 'input': 'racecar', 'expected': True},
                {'function': 'is_palindrome', 'input': 'hello', 'expected': False}
            ]
        },
        {
            'prompt': 'Write a Python function called to_uppercase that converts string to uppercase',
            'test_cases': [
                {'function': 'to_uppercase', 'input': 'hello', 'expected': 'HELLO'},
                {'function': 'to_uppercase', 'input': 'Test', 'expected': 'TEST'}
            ]
        },
        {
            'prompt': 'Write a Python function called count_words that counts words in a string',
            'test_cases': [
                {'function': 'count_words', 'input': 'hello world', 'expected': 2},
                {'function': 'count_words', 'input': 'one two three', 'expected': 3}
            ]
        },
        {
            'prompt': 'Write a Python function called first_char that returns first character of string',
            'test_cases': [
                {'function': 'first_char', 'input': 'hello', 'expected': 'h'},
                {'function': 'first_char', 'input': 'test', 'expected': 't'}
            ]
        },
        {
            'prompt': 'Write a Python function called string_length that returns length of string',
            'test_cases': [
                {'function': 'string_length', 'input': 'hello', 'expected': 5},
                {'function': 'string_length', 'input': 'test', 'expected': 4}
            ]
        },
        {
            'prompt': 'Write a Python function called remove_spaces that removes all spaces from string',
            'test_cases': [
                {'function': 'remove_spaces', 'input': 'hello world', 'expected': 'helloworld'},
                {'function': 'remove_spaces', 'input': 'a b c', 'expected': 'abc'}
            ]
        },
        {
            'prompt': 'Write a Python function called starts_with that checks if string starts with given prefix',
            'test_cases': [
                {'function': 'starts_with', 'input': ['hello', 'hel'], 'expected': True},
                {'function': 'starts_with', 'input': ['world', 'wor'], 'expected': True},
                {'function': 'starts_with', 'input': ['test', 'abc'], 'expected': False}
            ]
        },
        {
            'prompt': 'Write a Python function called repeat_string that repeats string n times',
            'test_cases': [
                {'function': 'repeat_string', 'input': ['hi', 3], 'expected': 'hihihi'},
                {'function': 'repeat_string', 'input': ['a', 4], 'expected': 'aaaa'}
            ]
        },
        {
            'prompt': 'Write a Python function called capitalize_first that capitalizes first letter',
            'test_cases': [
                {'function': 'capitalize_first', 'input': 'hello', 'expected': 'Hello'},
                {'function': 'capitalize_first', 'input': 'world', 'expected': 'World'}
            ]
        },
        {
            'prompt': 'Write a Python function called ends_with that checks if string ends with suffix',
            'test_cases': [
                {'function': 'ends_with', 'input': ['hello', 'lo'], 'expected': True},
                {'function': 'ends_with', 'input': ['test', 'st'], 'expected': True},
                {'function': 'ends_with', 'input': ['world', 'xyz'], 'expected': False}
            ]
        },
        {
            'prompt': 'Write a Python function called contains_digit that checks if string contains digit',
            'test_cases': [
                {'function': 'contains_digit', 'input': 'hello123', 'expected': True},
                {'function': 'contains_digit', 'input': 'hello', 'expected': False}
            ]
        },
        {
            'prompt': 'Write a Python function called to_lowercase that converts string to lowercase',
            'test_cases': [
                {'function': 'to_lowercase', 'input': 'HELLO', 'expected': 'hello'},
                {'function': 'to_lowercase', 'input': 'TeSt', 'expected': 'test'}
            ]
        },
        {
            'prompt': 'Write a Python function called replace_char that replaces character in string',
            'test_cases': [
                {'function': 'replace_char', 'input': ['hello', 'l', 'x'], 'expected': 'hexxo'},
                {'function': 'replace_char', 'input': ['test', 't', 'z'], 'expected': 'zesz'}
            ]
        },
        
        # List functions (15)
        {
            'prompt': 'Write a Python function called list_sum that returns sum of list',
            'test_cases': [
                {'function': 'list_sum', 'input': [[1, 2, 3]], 'expected': 6},
                {'function': 'list_sum', 'input': [[5, 10]], 'expected': 15}
            ]
        },
        {
            'prompt': 'Write a Python function called list_max that returns maximum in list',
            'test_cases': [
                {'function': 'list_max', 'input': [[1, 5, 3]], 'expected': 5},
                {'function': 'list_max', 'input': [[10, 2, 8]], 'expected': 10}
            ]
        },
        {
            'prompt': 'Write a Python function called list_min that returns minimum in list',
            'test_cases': [
                {'function': 'list_min', 'input': [[1, 5, 3]], 'expected': 1},
                {'function': 'list_min', 'input': [[10, 2, 8]], 'expected': 2}
            ]
        },
        {
            'prompt': 'Write a Python function called list_average that returns average of list',
            'test_cases': [
                {'function': 'list_average', 'input': [[2, 4, 6]], 'expected': 4.0},
                {'function': 'list_average', 'input': [[1, 2, 3, 4]], 'expected': 2.5}
            ]
        },
        {
            'prompt': 'Write a Python function called count_even that counts even numbers in list',
            'test_cases': [
                {'function': 'count_even', 'input': [[1, 2, 3, 4]], 'expected': 2},
                {'function': 'count_even', 'input': [[2, 4, 6]], 'expected': 3}
            ]
        },
        {
            'prompt': 'Write a Python function called remove_duplicates that removes duplicates from list',
            'test_cases': [
                {'function': 'remove_duplicates', 'input': [[1, 2, 2, 3]], 'expected': [1, 2, 3]},
                {'function': 'remove_duplicates', 'input': [[1, 1, 1]], 'expected': [1]}
            ]
        },
        {
            'prompt': 'Write a Python function called reverse_list that reverses a list',
            'test_cases': [
                {'function': 'reverse_list', 'input': [[1, 2, 3]], 'expected': [3, 2, 1]},
                {'function': 'reverse_list', 'input': [[5, 10]], 'expected': [10, 5]}
            ]
        },
        {
            'prompt': 'Write a Python function called list_contains that checks if list contains value',
            'test_cases': [
                {'function': 'list_contains', 'input': [[1, 2, 3], 2], 'expected': True},
                {'function': 'list_contains', 'input': [[1, 2, 3], 5], 'expected': False}
            ]
        },
        {
            'prompt': 'Write a Python function called list_length that returns length of list',
            'test_cases': [
                {'function': 'list_length', 'input': [[1, 2, 3]], 'expected': 3},
                {'function': 'list_length', 'input': [[1]], 'expected': 1}
            ]
        },
        {
            'prompt': 'Write a Python function called first_element that returns first element of list',
            'test_cases': [
                {'function': 'first_element', 'input': [[1, 2, 3]], 'expected': 1},
                {'function': 'first_element', 'input': [[5, 10]], 'expected': 5}
            ]
        },
        {
            'prompt': 'Write a Python function called last_element that returns last element of list',
            'test_cases': [
                {'function': 'last_element', 'input': [[1, 2, 3]], 'expected': 3},
                {'function': 'last_element', 'input': [[5, 10]], 'expected': 10}
            ]
        },
        {
            'prompt': 'Write a Python function called count_odd that counts odd numbers in list',
            'test_cases': [
                {'function': 'count_odd', 'input': [[1, 2, 3, 4]], 'expected': 2},
                {'function': 'count_odd', 'input': [[1, 3, 5]], 'expected': 3}
            ]
        },
        {
            'prompt': 'Write a Python function called list_product that returns product of all numbers in list',
            'test_cases': [
                {'function': 'list_product', 'input': [[2, 3, 4]], 'expected': 24},
                {'function': 'list_product', 'input': [[1, 5]], 'expected': 5}
            ]
        },
        {
            'prompt': 'Write a Python function called sort_list that sorts list in ascending order',
            'test_cases': [
                {'function': 'sort_list', 'input': [[3, 1, 2]], 'expected': [1, 2, 3]},
                {'function': 'sort_list', 'input': [[5, 2, 8, 1]], 'expected': [1, 2, 5, 8]}
            ]
        },
        {
            'prompt': 'Write a Python function called second_largest that returns second largest number in list',
            'test_cases': [
                {'function': 'second_largest', 'input': [[1, 5, 3]], 'expected': 3},
                {'function': 'second_largest', 'input': [[10, 2, 8]], 'expected': 8}
            ]
        },
        
        # Advanced functions (10)
        {
            'prompt': 'Write a Python function called factorial that calculates factorial',
            'test_cases': [
                {'function': 'factorial', 'input': 5, 'expected': 120},
                {'function': 'factorial', 'input': 3, 'expected': 6},
                {'function': 'factorial', 'input': 0, 'expected': 1}
            ]
        },
        {
            'prompt': 'Write a Python function called is_prime that checks if number is prime',
            'test_cases': [
                {'function': 'is_prime', 'input': 7, 'expected': True},
                {'function': 'is_prime', 'input': 4, 'expected': False},
                {'function': 'is_prime', 'input': 2, 'expected': True}
            ]
        },
        {
            'prompt': 'Write a Python function called fibonacci that returns nth fibonacci number',
            'test_cases': [
                {'function': 'fibonacci', 'input': 6, 'expected': 8},
                {'function': 'fibonacci', 'input': 7, 'expected': 13}
            ]
        },
        {
            'prompt': 'Write a Python function called gcd that finds greatest common divisor',
            'test_cases': [
                {'function': 'gcd', 'input': [48, 18], 'expected': 6},
                {'function': 'gcd', 'input': [100, 50], 'expected': 50}
            ]
        },
        {
            'prompt': 'Write a Python function called is_leap_year that checks if year is leap year',
            'test_cases': [
                {'function': 'is_leap_year', 'input': 2020, 'expected': True},
                {'function': 'is_leap_year', 'input': 2021, 'expected': False}
            ]
        },
        {
            'prompt': 'Write a Python function called binary_search that searches for value in sorted list',
            'test_cases': [
                {'function': 'binary_search', 'input': [[1, 2, 3, 4, 5], 3], 'expected': True},
                {'function': 'binary_search', 'input': [[1, 2, 3, 4, 5], 6], 'expected': False}
            ]
        },
        {
            'prompt': 'Write a Python function called bubble_sort that sorts list using bubble sort',
            'test_cases': [
                {'function': 'bubble_sort', 'input': [[3, 1, 4, 1, 5]], 'expected': [1, 1, 3, 4, 5]},
                {'function': 'bubble_sort', 'input': [[5, 2, 8]], 'expected': [2, 5, 8]}
            ]
        },
        {
            'prompt': 'Write a Python function called count_char that counts occurrences of character in string',
            'test_cases': [
                {'function': 'count_char', 'input': ['hello', 'l'], 'expected': 2},
                {'function': 'count_char', 'input': ['programming', 'm'], 'expected': 2}
            ]
        },
        {
            'prompt': 'Write a Python function called merge_lists that merges two sorted lists',
            'test_cases': [
                {'function': 'merge_lists', 'input': [[1, 3], [2, 4]], 'expected': [1, 2, 3, 4]},
                {'function': 'merge_lists', 'input': [[1, 5], [2, 3]], 'expected': [1, 2, 3, 5]}
            ]
        },
        {
            'prompt': 'Write a Python function called is_anagram that checks if two strings are anagrams',
            'test_cases': [
                {'function': 'is_anagram', 'input': ['listen', 'silent'], 'expected': True},
                {'function': 'is_anagram', 'input': ['hello', 'world'], 'expected': False}
            ]
        }
    ]
    
    return jsonify({'prompts': prompts})

if __name__ == '__main__':
    app.run(port=8080, debug=True)
