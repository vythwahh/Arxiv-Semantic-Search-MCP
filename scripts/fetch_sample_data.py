import sys
import json
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from embedder import ArxivEmbedder

TOPICS = [
    "large language models",
    "LLM reasoning",
    "LLM fine-tuning",
    "prompt engineering",
    "retrieval augmented generation"
]

def fetch_and_save(output_file: str = "scripts/data/sample_papers.json", max_per_topic: int = 150):
    embedder = ArxivEmbedder()
    all_papers = []
    seen_ids = set()

    for topic in TOPICS:
        print(f"\n🚀 Dang tien hanh cao chu de: '{topic}'...")
        papers = embedder.fetch_papers(query=topic, max_results=max_per_topic)

        for p in papers:
            if p.paper_id not in seen_ids:
                seen_ids.add(p.paper_id)
                all_papers.append({
                    "paper_id": p.paper_id,
                    "title": p.title,
                    "abstract": p.abstract,
                    "authors": p.authors,
                    "url": p.url,
                    "published": p.published,
                    "topic": topic
                })
        
        print(f"-> Gom duoc {len(papers)} bai cho '{topic}'. Nghi tay 3 giay...")
        time.sleep(3)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)

    print(f"\n XONNG ROII! Da luu tong cong {len(all_papers)} unique papers vao file {output_file}")

if __name__ == "__main__":
    fetch_and_save()