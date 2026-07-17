import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: str) -> List[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a stratified JSON subset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-label", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    emails = load_json(args.dataset)

    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for email in emails:
        by_label[str(email.get("label", ""))].append(email)

    subset: List[Dict[str, Any]] = []
    for label, items in sorted(by_label.items()):
        if len(items) < args.per_label:
            raise ValueError(
                f"Label {label!r} tem apenas {len(items)} exemplos; "
                f"pedido: {args.per_label}"
            )
        subset.extend(random.sample(items, args.per_label))

    random.shuffle(subset)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(subset, file, ensure_ascii=False, indent=2)

    print(f"{len(subset)} emails guardados em: {output_path}")
    print({label: args.per_label for label in sorted(by_label)})


if __name__ == "__main__":
    main()
