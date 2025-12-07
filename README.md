# Expense Tracker Application

A Streamlit-based expense tracking RAG application with data visualization and memory features.

## Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv venv
```

### 2. Activate Virtual Environment
```bash
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

### 5. Run the Application
```bash
streamlit run src/app.py
```

## Features

- **Streamlit UI** - Interactive web interface for expense tracking
- **Monthly Charts** - Visualize spending patterns per month
- **Memory Feature** - View your last question with "Show my last question"

## Demo

A recorded demonstration of the application is available in the `demo/` folder.


