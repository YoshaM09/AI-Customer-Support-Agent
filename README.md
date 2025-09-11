# Aven AI Customer Support Agent 🤖

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-4.9-blue)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-18.2-blue)](https://reactjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An AI-powered customer support agent for **Aven**, designed to answer user questions via **Natural Language** using a vector Database. Built with **Python, Pinecone, and TypeScript**, this project demonstrates the integration of web applications with AI-driven Q&A systems.

## ✨ Features

- **Vector Database**: Scrapes online information about Aven and stores it in a **Pinecone** vector database for fast semantic search.  
- **Natural Language Q&A**: Users can interact with the AI agent via a **TypeScript web app** supporting voice input.  
- **Accurate & Reliable Responses**: Designed to provide relevant answers by leveraging AI embeddings and similarity search.  
- **Optional Enhancements**:  
  - Evaluation set for measuring accuracy, helpfulness, and citation quality.  
  - Guardrails for handling sensitive queries (personal, legal, financial, or toxic content).  
  - Tool integration for scheduling meetings (optional).  

## 🛠 Tech Stack

- **Backend / AI**: Python, Gemini API , Pinecone API , Firecrawl API
- **Frontend**: TypeScript, React (or Next.js)  
- **Data Storage**: Pinecone vector database  
- **Others**: Vapi API for real-time chat  

## 🚀 Setup & Installation

### 1. **Clone the repository**  
```bash
git clone https://github.com/YoshaM09/AI-Customer-Support-Agent.git
cd AI-Customer-Support-Agent
```

### 2. **Install dependencies**
```bash
pip install -r requirements.txt
```
### 3. **Configure environment variables**
- .env file for API keys (OpenAI, Pinecone, etc.)

### 4. **Run the application**
- Backend: python main.py
- Frontend: npm run dev

## 🎬 Usage

- Open the web app in your browser.
- Interact with the AI agent via natural language.

## 🤝 Contributing

- Contributions are welcome! Please submit a pull request or open an issue for suggestions.

## 📄 License

- This project is licensed under the MIT License.
