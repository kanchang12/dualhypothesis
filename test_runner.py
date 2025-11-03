import requests
import json
import time
import sys

# Get URL from command line or use localhost
BASE_URL = sys.argv[1] if len(sys.argv) > 1 else 'http://localhost:8080'

print(f"Testing against: {BASE_URL}\n")

# 50 prompts - simple to complex
PROMPTS = [
    # Simple (15)
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
    
    # Medium (20)
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
    
    # Complex (15)
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

results = []

for i, prompt in enumerate(PROMPTS, 1):
    print(f"\n[{i}/50] Testing: {prompt[:50]}...")
    
    try:
        # Test Gemini only
        r1 = requests.post(f'{BASE_URL}/api/gemini', 
                          json={'prompt': prompt}, timeout=60)
        gemini_data = r1.json()
        
        time.sleep(1)
        
        # Test Gemini + OpenAI
        r2 = requests.post(f'{BASE_URL}/api/dual',
                          json={'prompt': prompt}, timeout=60)
        dual_data = r2.json()
        
        result = {
            'id': i,
            'prompt': prompt,
            'gemini_time': gemini_data['time'],
            'gemini_cost': gemini_data['cost'],
            'gemini_tokens': gemini_data['gemini_in'] + gemini_data['gemini_out'],
            'dual_time': dual_data['time'],
            'dual_cost': dual_data['total_cost'],
            'dual_tokens': dual_data['gemini_in'] + dual_data['gemini_out'] + dual_data['openai_in'] + dual_data['openai_out'],
            'cost_increase': dual_data['total_cost'] - gemini_data['cost'],
            'time_increase': dual_data['time'] - gemini_data['time']
        }
        
        results.append(result)
        
        print(f"  Gemini: {result['gemini_time']}s, ${result['gemini_cost']}")
        print(f"  Dual:   {result['dual_time']}s, ${result['dual_cost']}")
        print(f"  Diff:   +{result['time_increase']}s, +${result['cost_increase']:.6f}")
        
        time.sleep(2)  # Pause between tests
        
    except Exception as e:
        print(f"  ERROR: {e}")

# Save results
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
avg_gemini_time = sum(r['gemini_time'] for r in results) / len(results)
avg_dual_time = sum(r['dual_time'] for r in results) / len(results)
avg_gemini_cost = sum(r['gemini_cost'] for r in results) / len(results)
avg_dual_cost = sum(r['dual_cost'] for r in results) / len(results)
total_cost_increase = sum(r['cost_increase'] for r in results)

print(f"Tests completed: {len(results)}/50")
print(f"\nAverage Time:")
print(f"  Gemini: {avg_gemini_time:.2f}s")
print(f"  Dual:   {avg_dual_time:.2f}s")
print(f"  Increase: +{avg_dual_time - avg_gemini_time:.2f}s ({(avg_dual_time/avg_gemini_time - 1)*100:.1f}%)")
print(f"\nAverage Cost:")
print(f"  Gemini: ${avg_gemini_cost:.6f}")
print(f"  Dual:   ${avg_dual_cost:.6f}")
print(f"  Increase: +${avg_dual_cost - avg_gemini_cost:.6f}")
print(f"\nTotal extra cost for 50 tests: ${total_cost_increase:.4f}")
print("\nResults saved to results.json")
