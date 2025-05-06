# Italian Customs & Maritime Trade API

A simple FastAPI application that provides answers to questions about Italian customs procedures, maritime shipping, and international trade regulations.

## Overview

This API provides expert responses on topics including:
- Italian customs regulations and procedures
- Maritime shipping documentation
- Port operations and authorities
- Import/export requirements
- Customs duties and taxes

## Project Structure

- `main.py` - The FastAPI application that serves the Q&A functionality
- `navy_trade_data.jsonl` - Dataset of questions and expert answers
- `test_main.http` - Example HTTP requests for testing

## Getting Started

### Prerequisites

- Python 3.8+
- FastAPI
- Uvicorn (for serving the API)

### Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install fastapi uvicorn
   ```
3. Run the API server:
   ```
   uvicorn main:app --reload
   ```

The API will be available at http://127.0.0.1:8000

## API Usage

### Ask a Question

**Endpoint:** `POST /ask`

**Request Format:**
```json
{
  "question": "Cos'è il regime del deposito doganale?"
}
```