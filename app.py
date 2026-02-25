import os
import requests
import pandas as pd
from flask import Flask, request, jsonify
from groq import Groq

app = Flask(__name__)

# ==============================
# Environment Variables
# ==============================

MONDAY_API_KEY = os.getenv("MONDAY_API_KEY")
DEALS_BOARD_ID = os.getenv("DEALS_BOARD_ID")
WORK_BOARD_ID = os.getenv("WORK_BOARD_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

MONDAY_URL = "https://api.monday.com/v2"

# ==============================
# Fetch Board Data from Monday
# ==============================

def fetch_board_data(board_id):

    query = f"""
    query {{
      boards(ids: {board_id}) {{
        items_page(limit: 100) {{
          items {{
            name
            column_values {{
              column {{
                title
              }}
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

    response = requests.post(
        MONDAY_URL,
        json={"query": query},
        headers=headers
    )

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

# ==============================
# Health Check
# ==============================

@app.route("/")
def home():
    return "Monday BI Agent Running 🚀"

# ==============================
# Chat Endpoint (Groq LLM)
# ==============================

@app.route("/chat", methods=["POST"])
def chat():

    user_message = request.json.get("message")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    deals_df = fetch_board_data(DEALS_BOARD_ID)
    work_df = fetch_board_data(WORK_BOARD_ID)

    if deals_df.empty and work_df.empty:
        return jsonify({"error": "No data fetched from Monday boards"}), 500

    # Clean Deal Amount column if exists
    if "Deal Amount" in deals_df.columns:
        deals_df["Deal Amount"] = (
            deals_df["Deal Amount"]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("₹", "", regex=False)
        )
        deals_df["Deal Amount"] = pd.to_numeric(deals_df["Deal Amount"], errors="coerce")

    deals_summary = deals_df.head(100).to_string()
    work_summary = work_df.head(100).to_string()

    system_prompt = f"""
You are a senior Business Intelligence AI Agent.

You analyze live monday.com board data.

DEALS BOARD:
{deals_summary}

WORK ORDERS BOARD:
{work_summary}

Instructions:
- Give strategic business insights.
- Summarize patterns.
- Do not print raw tables.
- Be professional and concise.
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3
        )

        answer = response.choices[0].message.content

        return jsonify({"response": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)