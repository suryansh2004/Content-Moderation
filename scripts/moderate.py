from __future__ import annotations

import argparse
import sys

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send text to the content moderation API.")
    parser.add_argument("text", nargs="+", help="Text to moderate. Quote multi-word text.")
    parser.add_argument("--url", default="http://localhost:8002/moderate")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    text = " ".join(args.text)
    try:
        response = httpx.post(
            args.url,
            json={"texts": [text], "threshold": args.threshold},
            timeout=30,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1

    result = response.json()["results"][0]
    verdict = "FLAGGED" if result["flagged"] else "OK"
    print(f"{verdict} | max_score={result['max_score']:.4f}")
    for label in result["labels"]:
        marker = "*" if label["flagged"] else " "
        print(f"{marker} {label['label']}: {label['score']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
