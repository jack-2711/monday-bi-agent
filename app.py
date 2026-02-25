import os
import requests
import pandas as pd
from flask import Flask, request, jsonify
from groq import Groq

app = Flask(__name__)

# =========================
# Environment Variables
# =========================

MONDAY_API_KEY = os.getenv("MONDAY_API_KEY")
DEALS_BOARD_ID = os.getenv("DEALS_BOARD_ID")
WORK_BOARD_ID = os.getenv("WORK_BOARD_ID")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

MONDAY_URL = "https://api.monday.com/v2"


# =========================
# Fetch Board Data
# =========================

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

    if response.status_code != 200:
        return pd.DataFrame()

    data = response.json()

    items = data.get("data", {}).get("boards", [])[0].get("items_page", {}).get("items", [])

    rows = []
    for item in items:
        row = {"Item Name": item["name"]}
        for col in item["column_values"]:
            row[col["column"]["title"]] = col["text"]
        rows.append(row)

    df = pd.DataFrame(rows)

    # Try converting date columns safely
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col], errors="ignore")
        except:
            pass

    return df


# =========================
# Chat Endpoint
# =========================

@app.route("/")
def home():
    return "Monday BI Agent Running 🚀"


@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_message = request.json.get("message", "").lower()

        deals_df = fetch_board_data(DEALS_BOARD_ID)
        work_df = fetch_board_data(WORK_BOARD_ID)

        if deals_df.empty:
            return jsonify({"response": "No deal data available."})

        # Clean Deal Amount column
        if "Deal Amount" in deals_df.columns:
            deals_df["Deal Amount"] = (
                deals_df["Deal Amount"]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("₹", "", regex=False)
            )
            deals_df["Deal Amount"] = pd.to_numeric(deals_df["Deal Amount"], errors="coerce")

        # ============================
        # Mining Sector
        # ============================

        if "mining" in user_message:
            mining_deals = deals_df[
                deals_df.apply(
                    lambda row: row.astype(str).str.contains("Mining", case=False).any(),
                    axis=1
                )
            ]

            total_pipeline = mining_deals.get("Deal Amount", pd.Series()).sum()
            deal_count = len(mining_deals)

            insight_prompt = f"""
Mining Sector Pipeline Overview:

Total Deals: {deal_count}
Total Value: ₹{total_pipeline:,.2f}

Provide a short professional insight summary.
"""

        # ============================
        # Revenue / Pipeline
        # ============================

        elif "revenue" in user_message or "pipeline" in user_message:
            total_pipeline = deals_df.get("Deal Amount", pd.Series()).sum()
            deal_count = len(deals_df)

            insight_prompt = f"""
Overall Pipeline Overview:

Total Deals: {deal_count}
Total Value: ₹{total_pipeline:,.2f}

Provide a short professional business insight.
"""

        # ============================
        # Work Orders
        # ============================

        elif "work" in user_message or "execution" in user_message:
            if "Execution Status" in work_df.columns:
                status_counts = work_df["Execution Status"].value_counts().to_dict()
            else:
                status_counts = {}

            insight_prompt = f"""
Operational Execution Summary:

Status Breakdown:
{status_counts}

Provide a short operational insight.
"""

        else:
            return jsonify({
                "response": "Please ask about mining sector, pipeline, revenue, or work order performance."
            })

        # ============================
        # Groq LLM Call (Updated Model)
        # ============================

        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",,
            messages=[{"role": "user", "content": insight_prompt}],
            temperature=0.3
        )

        ai_reply = response.choices[0].message.content

        return jsonify({"response": ai_reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# Run
# =========================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
