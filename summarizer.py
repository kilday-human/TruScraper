import json, sys
from scraper import fetch_transcript
# If you want, replace with OpenAI or Claude bridge
from src.bridge import bridge_conversation  

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summarizer.py <YouTube_URL>")
        sys.exit(1)

    url = sys.argv[1]
    try:
        transcript = fetch_transcript(url)
    except Exception as e:
        print(f"❌ Error fetching transcript: {e}")
        sys.exit(1)

    with open("transcript.json", "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)

    text = " ".join(chunk["text"] for chunk in transcript)

    summary = bridge_conversation([
        {"role": "user", "content": f"Summarize this YouTube transcript:\n\n{text}"}
    ])

    with open("summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)

    print("\n--- SUMMARY ---\n")
    print(summary)
