"""
https://oai.azure.com/portal/be5567c3dd4d49eb93f58914cccf3f02/deployment
clausa gpt4
"""

import time
import requests
from dotenv import dotenv_values
import string
import requests
import uuid
import json
import asyncio
import aiohttp
from tqdm import tqdm
import concurrent.futures
from jinja2 import Template
import os
from dotenv import load_dotenv

load_dotenv()
def chatgpt(prompt, model, temperature=0.0, n=1, top_p=1, stop=None, max_tokens=1024, 
                  presence_penalty=0, frequency_penalty=0, logit_bias={}, timeout=30):
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "messages": messages,
        "model": model,
        "temperature": temperature,
        "n": n,
        "top_p": top_p,
        "stop": stop,
        "max_tokens": max_tokens,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias
    } 
    max_retries = 6
    base_delay = 1
    
    for retry in range(max_retries):
        try:
            r = requests.post('https://api.openai.com/v1/chat/completions',
                headers = {
                    "Authorization": f"Bearer {config['OPENAI_API_KEY']}",
                    "Content-Type": "application/json"
                },
                json = payload,
                timeout=timeout
            )
            
            if r.status_code == 200:
                r = r.json()
                return [choice['message']['content'] for choice in r['choices']]
            elif r.status_code == 429:  # Rate limit error
                retry_after = int(r.headers.get('Retry-After', base_delay * (2 ** retry)))
                print(f"Rate limit hit. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
            else:
                print(f"API Error - Status Code: {r.status_code}")
                if retry < max_retries - 1:
                    wait_time = base_delay * (2 ** retry)
                    time.sleep(wait_time)
                
        except requests.exceptions.RequestException:
            if retry < max_retries - 1:
                wait_time = base_delay * (2 ** retry)
                time.sleep(wait_time)
    
    raise Exception(f"Failed to get response after {max_retries} retries")


async def _fetch_single_completion(session, prompt, temperature, model, n, top_p, stop, max_tokens, presence_penalty, frequency_penalty, logit_bias, max_retries=8):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "n": n,
        "top_p": top_p,
        "stop": stop,
        "max_tokens": max_tokens,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "logit_bias": logit_bias
    }
    
    for attempt in range(max_retries):
        try:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status == 200:
                    response_json = await resp.json()
                    return response_json["choices"][0]["message"]["content"]
                elif resp.status == 429:  
                    retry_after = int(resp.headers.get('Retry-After', 5))
                    await asyncio.sleep(retry_after)
                    continue
                else:
                    error_text = await resp.text()
                    print(f"API Error - Status: {resp.status}, Error: {error_text}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt) 
                        continue
                    raise Exception(f"Failed single completion: {resp.status}, {error_text}")
        except asyncio.TimeoutError:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            raise
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            raise

async def _run_batch(prompts, temperature, model, n, top_p, stop, max_tokens, presence_penalty, frequency_penalty, logit_bias, timeout=60):
    connector = aiohttp.TCPConnector(limit=20)  
    timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            _fetch_single_completion(session, prompt, temperature, model, n, top_p, stop, max_tokens, presence_penalty, frequency_penalty, logit_bias)
            for prompt in prompts   
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

def chatgpt_batch(prompts, temperature, model='gpt-4o-mini', n=1, top_p=1, stop=None, max_tokens=1024, 
                  presence_penalty=0, frequency_penalty=0, logit_bias={}, timeout=60):
    try:
        results = asyncio.run(_run_batch(prompts, temperature, model, n, top_p, stop, max_tokens, presence_penalty, frequency_penalty, logit_bias, timeout))
        # Filter out any exceptions and return only successful results
        return [r for r in results if not isinstance(r, Exception)]
    except Exception as e:
        print(f"Batch processing failed: {str(e)}")

