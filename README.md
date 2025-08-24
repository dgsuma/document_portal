# Project Setup Guide

## Create Project Folder and Environment Setup

```bash
# Clone the repository (do this first)
git clone https://github.com/dgsuma/document_portal_multimodel_rag_app.git

# Move into the project folder
cd document_portal_multimodel_rag_app

# Open the folder in VS Code
code .

# (Optional) Create a new Conda environment in a folder called 'env'
conda create -p ./env python=3.10 -y

# Activate the environment (use the same path as above)
conda activate ./env

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Initialize Git (if not already initialized)
git init

# Stage all files
git add .

# Commit changes
git commit -m "<write your commit message>"  # Replace with your commit message

# Push to remote (after adding remote origin)
git push
```
## Minimum Requirements for the Project

### LLM Models
- **Groq** (Free)
- **OpenAI** (Paid)
- **Gemini** (15 Days Free Access)
- **Claude** (Paid)
- **Hugging Face** (Free)
- **Ollama** (Local Setup)

### Embedding Models
- **OpenAI**
- **Hugging Face**
- **Gemini**

### Vector Databases
- **In-Memory**
- **On-Disk**
- **Cloud-Based**

## API Keys

### GROQ API Key
- [Get your API Key](https://console.groq.com/keys)  
- [Groq Documentation](https://console.groq.com/docs/overview)

### Gemini API Key
- [Get your API Key](https://aistudio.google.com/apikey)  
- [Gemini Documentation](https://ai.google.dev/gemini-api/docs/models)
### Gemini API Key
- [Get your API Key](https://aistudio.google.com/apikey)  
- [Gemini Documentation](https://ai.google.dev/gemini-api/docs/models)


