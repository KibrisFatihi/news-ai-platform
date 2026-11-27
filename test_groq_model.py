"""Groq model erişim testi - tek komutla çalıştırılabilir."""
import requests


def main():
    key = input("Groq API anahtarın: ").strip()
    if not key.startswith("gsk_"):
        print("Uyarı: anahtar gsk_ ile başlamıyor olabilir.")

    models = [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "gemma2-9b-it",
        "mixtral-8x7b-32768",
    ]

    print("\nModel denemesi başlıyor...\n")
    for m in models:
        try:
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": "Bearer " + key},
                json={"model": m, "messages": [{"role": "user", "content": "selam"}], "max_tokens": 10},
                timeout=10,
            )
            print(f"{m:30s} -> {r.status_code} {r.text[:70].replace(chr(10), ' ')}")
        except Exception as e:
            print(f"{m:30s} -> HATA: {e}")

    print("\n200 dönen modeli config.py'deki default_model yapabilirsin.")


if __name__ == "__main__":
    main()
