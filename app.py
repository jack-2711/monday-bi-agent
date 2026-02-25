import os
import requests
import pandas as pd
from flask import Flask, request, jsonify
from groq import Groq

app = Flask(__name__)

MONDAY_API_KEY = os.getenv("MONDAY_API_KEY")
DEALS_BOARD_ID = os.getenv("DEALS_BOARD_ID")
WORK_BOARD_ID = os.getenv("WORK_BOARD_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

MONDAY_URL = "https://api.monday.com/v2"

def fetch_board_data(board_id):
    query = f"""
    query {{
      boards(ids: {int(board_id)}) {{
        items_page(limit: 100) {{
          items {{
            name
            column_values {{
              column {{ title }}
              text
            }}
          }}
        }}
      }}
    }}
    """

    headers = {
        "Authorization": MONDAY_API_KEY,
        "Content-Type": "application/json"
    }

    response = requests.post(MONDAY_URL, json={"query": query}, headers=headers)
    data = response.json()

    try:
        items = data["data"]["boards"][0]["items_page"]["items"]
    except:
        return pd.DataFrame()

    rows = []

    for item in items:
        row = {"Item": item["name"]}
        for col in item["column_values"]:
            row[col["column"]["title"]] = col["text"]
        rows.append(row)

    return pd.DataFrame(rows)


@app.route("/")
def home():
    return "Monday BI Agent Running 🚀"


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")

    deals_df = fetch_board_data(DEALS_BOARD_ID)
    work_df = fetch_board_data(WORK_BOARD_ID)

    deals_summary = deals_df.head(50).to_string()
    work_summary = work_df.head(50).to_string()

    system_prompt = f"""
You are a Business Intelligence AI Agent.

DEALS BOARD:
{deals_summary}

WORK ORDERS BOARD:
{work_summary}

Provide strategic founder-level insights.
Do not dump raw tables.
Be concise and professional.
"""

    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
        temperature=0.3
    )

    return jsonify({
        "response": response.choices[0].message.content
    })


if __name__ == "__main__":
    app.run(debug=True)