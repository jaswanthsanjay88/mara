"""
Performs exhaustive train/test overlap and nearest-neighbor similarity analysis
between the 1,650 domain training prompts (data/needle_train.jsonl) and the
250 frozen test prompts (data/frozen_eval_250.jsonl).
Computes:
1. Normalized string exact collisions.
2. Token Jaccard similarity nearest neighbors.
3. Character SequenceMatcher similarity nearest neighbors.
4. Per-category similarity distributions (clean, paraphrase, typo_noise, compound, hard_negative).
"""

import difflib
import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(ROOT, "data", "needle_train.jsonl")
TEST_PATH = os.path.join(ROOT, "data", "frozen_eval_250.jsonl")
OUT_PATH = os.path.join(ROOT, "data", "train_test_overlap_analysis.json")


def normalize_text(text: str) -> str:
    t = text.lower().strip()
    t = re.sub(r"[^\w\s]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t


def jaccard(s1: str, s2: str) -> float:
    set1 = set(s1.split())
    set2 = set(s2.split())
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def main():
    print("=" * 80)
    print("COMPUTING TRAIN-TEST OVERLAP & NEAREST-NEIGHBOR SIMILARITY MATRIX")
    print("=" * 80)

    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f if line.strip()]
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f if line.strip()]

    train_queries = [item["query"] for item in train_data]
    train_norm = [normalize_text(q) for q in train_queries]

    # Exact normalized collisions
    exact_collisions = []
    category_stats = defaultdict(lambda: {"jaccards": [], "seq_ratios": [], "samples": []})

    all_jaccards = []
    all_seq_ratios = []

    for idx, test_item in enumerate(test_data):
        q = test_item["query"]
        cat = test_item["category"]
        q_norm = normalize_text(q)

        # Check exact collision
        if q_norm in train_norm:
            matched_train = train_queries[train_norm.index(q_norm)]
            exact_collisions.append({
                "test_idx": idx,
                "category": cat,
                "test_query": q,
                "train_query": matched_train
            })

        # Nearest neighbor search
        best_j = 0.0
        best_seq = 0.0
        best_train_q = ""

        for t_idx, t_norm in enumerate(train_norm):
            j = jaccard(q_norm, t_norm)
            if j > best_j:
                best_j = j
                best_train_q = train_queries[t_idx]
            if j > 0.5:
                seq = difflib.SequenceMatcher(None, q_norm, t_norm).ratio()
                if seq > best_seq:
                    best_seq = seq

        if best_seq == 0.0 and best_train_q:
            best_seq = difflib.SequenceMatcher(None, q_norm, normalize_text(best_train_q)).ratio()

        all_jaccards.append(best_j)
        all_seq_ratios.append(best_seq)

        category_stats[cat]["jaccards"].append(best_j)
        category_stats[cat]["seq_ratios"].append(best_seq)
        category_stats[cat]["samples"].append({
            "test_query": q,
            "nearest_train_query": best_train_q,
            "jaccard": round(best_j, 3),
            "seq_ratio": round(best_seq, 3),
        })

    def calc_dist(vals: List[float]) -> Dict[str, float]:
        if not vals:
            return {}
        sv = sorted(vals)
        return {
            "mean": round(sum(vals) / len(vals), 3),
            "median": round(sv[len(sv) // 2], 3),
            "p90": round(sv[int(len(sv) * 0.90)], 3),
            "p95": round(sv[int(len(sv) * 0.95)], 3),
            "max": round(max(vals), 3),
            "min": round(min(vals), 3),
        }

    overall_jaccard_dist = calc_dist(all_jaccards)
    overall_seq_dist = calc_dist(all_seq_ratios)

    per_category_analysis = {}
    for cat, data in category_stats.items():
        # Sort samples by jaccard descending
        sorted_samples = sorted(data["samples"], key=lambda x: x["jaccard"], reverse=True)
        per_category_analysis[cat] = {
            "sample_count": len(data["jaccards"]),
            "jaccard_distribution": calc_dist(data["jaccards"]),
            "seq_ratio_distribution": calc_dist(data["seq_ratios"]),
            "top_5_closest_pairs": sorted_samples[:5],
        }

    analysis_result = {
        "train_samples_count": len(train_queries),
        "test_samples_count": len(test_data),
        "exact_normalized_collisions": {
            "count": len(exact_collisions),
            "rate_percent": round(len(exact_collisions) / len(test_data) * 100.0, 2),
            "items": exact_collisions,
        },
        "overall_jaccard_distribution": overall_jaccard_dist,
        "overall_sequence_ratio_distribution": overall_seq_dist,
        "per_category_analysis": per_category_analysis,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(analysis_result, f, indent=2)

    print(f"\n[+] Analysis complete!")
    print(f"    Exact Normalized Collisions: {len(exact_collisions)} / {len(test_data)} ({len(exact_collisions)/len(test_data)*100:.1f}%)")
    print(f"    Nearest-Neighbor Jaccard:    Mean={overall_jaccard_dist['mean']}, Median={overall_jaccard_dist['median']}, Max={overall_jaccard_dist['max']}")
    print(f"    Nearest-Neighbor SeqRatio:   Mean={overall_seq_dist['mean']}, Median={overall_seq_dist['median']}, Max={overall_seq_dist['max']}")
    print(f"\n    Per-Category Mean Jaccard:")
    for cat in ["clean", "paraphrase", "typo_noise", "compound", "hard_negative"]:
        print(f"      - {cat:<15}: Jaccard Mean={per_category_analysis[cat]['jaccard_distribution']['mean']} | Max={per_category_analysis[cat]['jaccard_distribution']['max']}")

    print(f"\n[+] Detailed report written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
