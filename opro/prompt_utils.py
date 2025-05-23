# Copyright 2023 The OPRO Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The utility functions for prompting GPT and Google Cloud models."""

import time
import requests
import google.generativeai as palm
import openai
import os
from dotenv import load_dotenv
import opro.utils_mr

load_dotenv()  # Load environment variables from .env file

def call_openai_server_single_prompt(prompt, model, max_decode_steps=1024, temperature=0.0, n=1, top_p=1, stop=None,
                  presence_penalty=0, frequency_penalty=0, logit_bias={}, timeout=10):
    messages = [{"role": "user", "content": prompt}]
    payload = {
        "messages": messages,
        "model": model,
        "temperature": temperature,
        "n": n,
        "top_p": top_p,
        "stop": stop,
        "max_tokens": max_decode_steps,
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
                    "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
                    "Content-Type": "application/json"
                },
                json = payload,
                timeout=timeout
            )
            
            if r.status_code == 200:
                r = r.json()
                print(f"r: {r['choices'][0]['message']['content']}")
                return r['choices'][0]['message']['content']
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

# def call_openai_server_single_prompt(
#     prompt, model="gpt-4o-mini", max_decode_steps=20, temperature=0.8
# ):
#   """The function to call OpenAI server with an input string."""
#   try:
#     completion = openai.ChatCompletion.create(
#         model=model,
#         temperature=temperature,
#         max_tokens=max_decode_steps,
#         messages=[
#             {"role": "user", "content": prompt},
#         ],
#     )
#     return completion.choices[0].message.content

#   except openai.error.Timeout as e:
#     retry_time = e.retry_after if hasattr(e, "retry_after") else 30
#     print(f"Timeout error occurred. Retrying in {retry_time} seconds...")
#     time.sleep(retry_time)
#     return call_openai_server_single_prompt(
#         prompt, max_decode_steps=max_decode_steps, temperature=temperature
#     )

#   except openai.error.RateLimitError as e:
#     retry_time = e.retry_after if hasattr(e, "retry_after") else 30
#     print(f"Rate limit exceeded. Retrying in {retry_time} seconds...")
#     time.sleep(retry_time)
#     return call_openai_server_single_prompt(
#         prompt, max_decode_steps=max_decode_steps, temperature=temperature
#     )

#   except openai.error.APIError as e:
#     retry_time = e.retry_after if hasattr(e, "retry_after") else 30
#     print(f"API error occurred. Retrying in {retry_time} seconds...")
#     time.sleep(retry_time)
#     return call_openai_server_single_prompt(
#         prompt, max_decode_steps=max_decode_steps, temperature=temperature
#     )

#   except openai.error.APIConnectionError as e:
#     retry_time = e.retry_after if hasattr(e, "retry_after") else 30
#     print(f"API connection error occurred. Retrying in {retry_time} seconds...")
#     time.sleep(retry_time)
#     return call_openai_server_single_prompt(
#         prompt, max_decode_steps=max_decode_steps, temperature=temperature
#     )

#   except openai.error.ServiceUnavailableError as e:
#     retry_time = e.retry_after if hasattr(e, "retry_after") else 30
#     print(f"Service unavailable. Retrying in {retry_time} seconds...")
#     time.sleep(retry_time)
#     return call_openai_server_single_prompt(
#         prompt, max_decode_steps=max_decode_steps, temperature=temperature
#     )

#   except OSError as e:
#     retry_time = 5  # Adjust the retry time as needed
#     print(
#         f"Connection error occurred: {e}. Retrying in {retry_time} seconds..."
#     )
#     time.sleep(retry_time)
#     return call_openai_server_single_prompt(
#         prompt, max_decode_steps=max_decode_steps, temperature=temperature
#     )


def call_openai_server_func(
    inputs, model="gpt-4o-mini", max_decode_steps=1024, temperature=0.8
):
  """The function to call OpenAI server with a list of input strings."""
  outputs = []
  if isinstance(inputs, str):
    inputs = [inputs]
 
  if (len(inputs) > 1):
    outputs = opro.utils_mr.chatgpt_batch(
      inputs,
      temperature=temperature,
      model=model,
      max_tokens=max_decode_steps,
    )
  else:
    outputs = call_openai_server_single_prompt(
      inputs[0],
      model=model,
      max_decode_steps=max_decode_steps,
      temperature=temperature,
    )
  return outputs


def call_palm_server_from_cloud(
    input_text, model="text-bison-001", max_decode_steps=20, temperature=0.8
):
  """Calling the text-bison model from Cloud API."""
  assert isinstance(input_text, str)
  assert model == "text-bison-001"
  all_model_names = [
      m
      for m in palm.list_models()
      if "generateText" in m.supported_generation_methods
  ]
  model_name = all_model_names[0].name
  try:
    completion = palm.generate_text(
        model=model_name,
        prompt=input_text,
        temperature=temperature,
        max_output_tokens=max_decode_steps,
    )
    output_text = completion.result
    return [output_text]
  except:  # pylint: disable=bare-except
    retry_time = 10  # Adjust the retry time as needed
    print(f"Retrying in {retry_time} seconds...")
    time.sleep(retry_time)
    return call_palm_server_from_cloud(
        input_text, max_decode_steps=max_decode_steps, temperature=temperature
    )
