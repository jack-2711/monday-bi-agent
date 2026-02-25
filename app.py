import os
import json
import pandas as pd
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

############################################
# 1️⃣ LOAD + CLEAN DATA
############################################

def load_data():
    try:
        deals_df = pd.read_csv("Deal_funnel_Data.csv")
        work_df = pd.read_csv("Work_Order_Tracker_Data.csv")

        deals_df = clean_df(deals_df)
        work_df = clean_df(work_df)

        return deals_df, work_df

    except Exception as e:
        print("Data loading error:", e)
        return None, None


def clean_df(df):
    df = df.fillna("Unknown")

    for col in df.columns:

        # Normalize numeric columns
        if any(word in col.lower() for word in ["value", "revenue", "amount"]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Normalize date columns
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")

        # Normalize text
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().str.lower()

    return df


############################################
# 2️⃣ QUERY PARSER
############################################

def parse_query(user_query):
    try:
        prompt = f"""
        Extract structured meaning from this founder-level business question.

        Question: {user_query}

        Return ONLY valid JSON:
        {{
          "sector": null or string,
          "metric": string
        }}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.choices[0].message.content
        return json.loads(content)

    except Exception as e:
        print("Parsing error:", e)
        return {"sector": None, "metric": "general"}


############################################
# 3️⃣ BUSINESS LOGIC
############################################

def analyze_data(user_query, deals_df, work_df):

    parsed = parse_query(user_query)
    sector = parsed.get("sector")

    # Filter by sector if mentioned
    if sector:
        deals_df = deals_df[
            deals_df.apply(lambda row: row.astype(str).str.contains(sector).any(), axis=1)
        ]
        work_df = work_df[
            work_df.apply(lambda row: row.astype(str).str.contains(sector).any(), axis=1)
        ]

    total_pipeline = deals_df.select_dtypes(include='number').sum().sum()
    total_revenue = work_df.select_dtypes(include='number').sum().sum()

    deal_count = len(deals_df)
    work_count = len(work_df)

    missing_values = deals_df.isnull().sum().sum() + work_df.isnull().sum().sum()

    summary = f"""
    Sector Filter: {sector}
    Total pipeline value: {total_pipeline}
    Total executed revenue: {total_revenue}
    Number of deals: {deal_count}
    Number of work orders: {work_count}
    Missing values detected: {missing_values}
    """

    insight_prompt = f"""
    You are a strategic business advisor.

    Founder Question:
    {user_query}

    Business Data Summary:
    {summary}

    Provide:
    - Direct answer
    - Strategic insight
    - Risks
    - Mention data quality caveats
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": insight_prompt}]
    )

    return response.choices[0].message.content


############################################
# 4️⃣ CHAT ENDPOINT
############################################

@app.route("/chat", methods=["POST"])
def chat():
    user_query = request.json.get("message")

    if not user_query:
        return jsonify({"error": "No message provided"}), 400

    deals_df, work_df = load_data()

    if deals_df is None:
        return jsonify({"error": "Data loading failed"}), 500

    result = analyze_data(user_query, deals_df, work_df)

    return jsonify({"response": result})


############################################
# 5️⃣ ROOT CHECK
############################################

@app.route("/")
def home():
    return "Monday BI Agent Running 🚀"


############################################
# RUN
############################################

if __name__ == "__main__":
    app.run(debug=True)