import requests

def test_telegram(token, chat_id, text):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, data=data, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response Body: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    TOKEN = "8587616990:AAEEBekE6eY8xVdFL1We43Vsw7xIXpjucew"
    # Probando quitando el 100 inicial y poniendo el prefijo -100
    CHAT_ID = "-1003591233672"
    test_telegram(TOKEN, CHAT_ID, "Segunda prueba con ID ajustado desde Visión AI 🚀")
