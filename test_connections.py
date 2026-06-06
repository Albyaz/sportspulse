import os
import requests
from dotenv import load_dotenv
import psycopg2

load_dotenv()

# Test 1 — Environment variables
print("=== Testing Environment Variables ===")
football_key = os.getenv("FOOTBALL_API_KEY")
news_key = os.getenv("NEWS_API_KEY")
db_url = os.getenv("DATABASE_URL")

print(f"Football API Key: {'✓ Found' if football_key else '✗ Missing'}")
print(f"News API Key:     {'✓ Found' if news_key else '✗ Missing'}")
print(f"Database URL:     {'✓ Found' if db_url else '✗ Missing'}")

# Test 2 — API-Football
print("\n=== Testing API-Football ===")
try:
    headers = {"x-apisports-key": football_key}
    response = requests.get("https://v3.football.api-sports.io/status", headers=headers)
    data = response.json()
    account = data["response"]["account"]
    requests_info = data["response"]["requests"]
    print(f"✓ Connected — {account['firstname']} {account['lastname']}")
    print(f"  Plan: {data['response']['subscription']['plan']}")
    print(f"  Requests today: {requests_info['current']} / {requests_info['limit_day']}")
except Exception as e:
    print(f"✗ API-Football error: {e}")

# Test 3 — NewsAPI
print("\n=== Testing NewsAPI ===")
try:
    url = f"https://newsapi.org/v2/top-headlines?category=sports&apiKey={news_key}&pageSize=1"
    response = requests.get(url)
    if response.status_code == 200:
        print(f"✓ NewsAPI connected")
    else:
        print(f"✗ NewsAPI failed — Status: {response.status_code}")
except Exception as e:
    print(f"✗ NewsAPI error: {e}")

# Test 4 — PostgreSQL
print("\n=== Testing PostgreSQL (Neon) ===")
try:
    conn = psycopg2.connect(db_url)
    cursor = conn.cursor()
    cursor.execute("SELECT version();")
    version = cursor.fetchone()
    print(f"✓ PostgreSQL connected — {version[0][:50]}")
    conn.close()
except Exception as e:
    print(f"✗ PostgreSQL error: {e}")

print("\n=== All Connection Tests Complete ===")