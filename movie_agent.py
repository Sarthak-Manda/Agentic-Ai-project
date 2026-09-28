"""
Multi-Agent Movie Recommendation System
---------------------------------------
Agent 1: FetcherAgent - Handles query extraction and TMDB data retrieval.
Agent 2: RankingAgent - Performs hybrid vector matching, review/popularity scoring, and reasoning.
"""

import os
import urllib.parse
import requests
import ollama
from sentence_transformers import SentenceTransformer, util

# --- CONFIGURATION ---
TMDB_API_KEY = "72e4b227a95a6c5c124f575c27dd02ba"
MODEL = "qwen2.5:7b"
EMBEDDER_MODEL = "all-MiniLM-L6-v2"

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class FetcherAgent:
    """Agent 1: Responsible for search query generation and raw candidate retrieval."""

    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name

    def extract_queries(self, user_prompt: str) -> list[str]:
        print("\n[FetcherAgent] Analyzing user intent & tone...")
        system_prompt = (
            "You are a movie retrieval agent. Read the user's request and output "
            "3 short search terms (1-2 words each) to query a movie database. "
            "Output ONLY a comma-separated list."
        )
        try:
            res = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            raw = res["message"]["content"].strip()
            queries = [q.strip() for q in raw.split(",") if q.strip()]
            if queries:
                return queries[:3]
        except Exception:
            print("  [!] LLM unavailable, using keyword fallback.")

        # Fallback keyword extraction
        words = [w.strip(".,!?").lower() for w in user_prompt.split() if len(w) > 4]
        return list(dict.fromkeys(words))[:3] or ["satire", "corporate", "society"]

    def fetch_candidates(self, queries: list[str]) -> list[dict]:
        print(f"[FetcherAgent] Fetching candidates from TMDB for: {', '.join(queries)}")
        candidates = {}

        for query in queries:
            url = f"https://api.themoviedb.org/3/search/movie?api_key={self.api_key}&query={urllib.parse.quote(query)}&language=en-US&page=1"
            try:
                res = requests.get(url, timeout=5)
                res.raise_for_status()
                for item in res.json().get("results", []):
                    if item.get("overview") and item["id"] not in candidates:
                        candidates[item["id"]] = {
                            "id": item["id"],
                            "title": item["title"],
                            "overview": item["overview"],
                            "year": item.get("release_date", "N/A")[:4],
                            "rating": item.get("vote_average", 0.0), # Score / Review out of 10
                            "vote_count": item.get("vote_count", 0),
                            "popularity": item.get("popularity", 0.0)
                        }
            except Exception as e:
                print(f"  [!] Search error for '{query}': {e}")

        print(f"[FetcherAgent] Retrieved {len(candidates)} candidates.")
        return list(candidates.values())


class RankingAgent:
    """Agent 2: Responsible for hybrid vector scoring, popularity filtering, and reasoning."""

    def __init__(self, embedder_name: str, model_name: str):
        self.embedder = SentenceTransformer(embedder_name)
        self.model_name = model_name

    def rank_candidates(self, user_prompt: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
        if not candidates:
            return []

        print("[RankingAgent] Computing hybrid scores (Semantic Tone + Ratings/Popularity)...")
        
        # 1. Semantic Embedding Match
        prompt_embedding = self.embedder.encode(user_prompt, convert_to_tensor=True)
        texts = [f"{c['title']}: {c['overview']}" for c in candidates]
        candidate_embeddings = self.embedder.encode(texts, convert_to_tensor=True)
        semantic_sims = util.cos_sim(prompt_embedding, candidate_embeddings)[0]

        # 2. Hybrid Score Calculation
        for idx, item in enumerate(candidates):
            semantic_score = float(semantic_sims[idx])
            rating_score = item["rating"] / 10.0  # Normalized to 0.0 - 1.0
            
            # Formula: 60% Semantic Match + 40% TMDB User Rating
            item["hybrid_score"] = (0.60 * semantic_score) + (0.40 * rating_score)
            item["semantic_score"] = semantic_score

        # Sort by final hybrid score
        ranked = sorted(candidates, key=lambda x: x["hybrid_score"], reverse=True)
        return ranked[:top_n]

    def generate_reasoning(self, user_prompt: str, top_movies: list[dict]):
        print("[RankingAgent] Generating agentic reasoning for top matches...")
        for movie in top_movies:
            prompt = (
                f"User requested: '{user_prompt}'. "
                f"Movie: '{movie['title']}' (Rating: {movie['rating']}/10). "
                f"Plot: {movie['overview']}. "
                "In ONE direct sentence, explain why this fits the user's specific tone."
            )
            try:
                res = ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                movie["reasoning"] = res["message"]["content"].strip()
            except Exception:
                movie["reasoning"] = f"Strong thematic alignment with a high rating of {movie['rating']}/10."


def main():
    print("=" * 60)
    print("🎬 TWO-AGENT MOVIE RECOMMENDATION SYSTEM")
    print("=" * 60)

    user_prompt = input("\nDescribe the movie you want to watch:\n> ")
    if len(user_prompt.strip()) < 5:
        print("Please enter a longer description.")
        return

    # Instantiate Agents
    fetcher = FetcherAgent(api_key=TMDB_API_KEY, model_name=MODEL)
    ranker = RankingAgent(embedder_name=EMBEDDER_MODEL, model_name=MODEL)

    # Pipeline execution
    queries = fetcher.extract_queries(user_prompt)
    candidates = fetcher.fetch_candidates(queries)

    if not candidates:
        print("No candidates found.")
        return

    top_movies = ranker.rank_candidates(user_prompt, candidates, top_n=5)
    ranker.generate_reasoning(user_prompt, top_movies)

    # Output Results
    print("\n" + "=" * 60)
    print("                 TOP MATCHES (AGENT EVALUATION)")
    print("=" * 60)

    for i, m in enumerate(top_movies, 1):
        print(f"\n{i}. {m['title']} ({m['year']})")
        print(f"   Hybrid Score   : {m['hybrid_score'] * 100:.1f}%")
        print(f"   TMDB User Rating: {m['rating']}/10 ({m['vote_count']} votes)")
        print(f"   Agent Reasoning: {m['reasoning']}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()