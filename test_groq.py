import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

try:
    client = Groq(api_key=os.getenv('GROQ_API_KEY'))
    print("Groq Client init successful.")
    
    for model in ["llama-3.3-70b-versatile", "deepseek-r1-distill-llama-70b"]:
        try:
            print(f"Testing {model}...")
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say hi!"}],
                max_tokens=10
            )
            print(f"Success! {model} responded: {resp.choices[0].message.content}")
        except Exception as e:
            print(f"FAILED {model}: {e}")
            
except Exception as e:
    print(f"Init failed: {e}")
