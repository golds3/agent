from langchain_ollama import OllamaLLM
 
llm = OllamaLLM(model="deepseek-r1:7b")

def getLLM():
    return llm