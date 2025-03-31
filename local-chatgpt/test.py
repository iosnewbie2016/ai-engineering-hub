import ollama

response_generator = ollama.chat(model="mistral", messages=[{"role": "user", "content": "Hello"}], stream=True)
print(type(response_generator))
