import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


QUESTION_BY_CATEGORY = {
    "participants": "Quem participa na reuniao?",
    "time": "Quando e a reuniao?",
    "time_normalized": "Qual e a data/hora normalizada da reuniao?",
    "location": "Onde e a reuniao?",
    "topic": "Qual e o topico da reuniao?",
}

DEFAULT_CATEGORIES = [
    "participants",
    "time",
    "time_normalized",
    "location",
    "topic",
]


def as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def answer_for_category(arguments: Dict[str, Any], category: str) -> Optional[str]:
    values = as_list(arguments.get(category))
    if not values:
        return None
    return "; ".join(values)


def build_context(item: Dict[str, Any], categories: Iterable[str]) -> str:
    arguments = item.get("arguments", {})
    lines = [
        f"Assunto: {item.get('subject', '')}",
        f"Data de envio: {item.get('sent_datetime') or ''}",
        "",
        str(item.get("text", "") or "").strip(),
        "",
        "Informacao estruturada:",
    ]

    labels = {
        "participants": "Participantes",
        "time": "Data/hora textual",
        "time_normalized": "Data/hora normalizada",
        "location": "Localizacao",
        "topic": "Topico",
    }
    for category in categories:
        answer = answer_for_category(arguments, category)
        if answer:
            lines.append(f"{labels[category]}: {answer}")

    return "\n".join(lines).strip()


def make_example(
    item: Dict[str, Any],
    category: str,
    include_impossible: bool,
) -> Optional[Dict[str, Any]]:
    arguments = item.get("arguments", {})
    answer = answer_for_category(arguments, category)
    if not answer and not include_impossible:
        return None

    context = build_context(item, DEFAULT_CATEGORIES)
    answer_start = context.find(answer) if answer else -1
    if answer and answer_start < 0:
        raise ValueError(
            f"Resposta nao encontrada no contexto: id={item.get('id')} "
            f"category={category} answer={answer!r}"
        )

    example_id = f"{item.get('id')}:{category}"
    return {
        "id": example_id,
        "title": str(item.get("subject", "") or ""),
        "context": context,
        "question": QUESTION_BY_CATEGORY[category],
        "answers": {
            "text": [answer] if answer else [],
            "answer_start": [answer_start] if answer else [],
        },
        "category": category,
        "email_id": item.get("id"),
        "dataset_id": item.get("dataset_id"),
        "intent": item.get("intent"),
        "sent_datetime": item.get("sent_datetime"),
        "is_impossible": not bool(answer),
    }


def convert_gold(
    gold: List[Dict[str, Any]],
    categories: List[str],
    include_impossible: bool,
) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    unknown_categories = sorted(set(categories) - set(QUESTION_BY_CATEGORY))
    if unknown_categories:
        raise ValueError(f"Categorias desconhecidas: {unknown_categories}")

    for item in gold:
        for category in categories:
            example = make_example(item, category, include_impossible)
            if example:
                examples.append(example)
    return examples


def split_examples(
    examples: List[Dict[str, Any]],
    validation_size: float,
    test_size: float,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    if validation_size < 0 or test_size < 0 or validation_size + test_size >= 1:
        raise ValueError("validation_size + test_size deve estar entre 0 e 1.")

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for example in examples:
        key = str(example.get("email_id"))
        grouped.setdefault(key, []).append(example)

    rng = random.Random(seed)
    email_ids = list(grouped)
    rng.shuffle(email_ids)

    total_groups = len(email_ids)
    test_group_count = int(round(total_groups * test_size))
    validation_group_count = int(round(total_groups * validation_size))

    test_ids = set(email_ids[:test_group_count])
    validation_ids = set(email_ids[test_group_count:test_group_count + validation_group_count])

    train: List[Dict[str, Any]] = []
    validation: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []
    for email_id, group in grouped.items():
        if email_id in test_ids:
            test.extend(group)
        elif email_id in validation_ids:
            validation.extend(group)
        else:
            train.extend(group)

    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return {
        "train": train,
        "validation": validation,
        "test": test,
    }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            json.dump(row, file, ensure_ascii=False)
            file.write("\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def category_counts(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        category = str(row.get("category"))
        counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Converte gold annotations para dataset HuggingFace QA JSONL."
    )
    parser.add_argument("--gold", required=True, help="Ficheiro gold JSON.")
    parser.add_argument("--output-dir", required=True, help="Diretorio de output.")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="Categorias a converter.",
    )
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-impossible",
        action="store_true",
        help="Inclui perguntas sem resposta. Por defeito sao ignoradas.",
    )
    args = parser.parse_args()

    gold_path = Path(args.gold)
    output_dir = Path(args.output_dir)
    with gold_path.open("r", encoding="utf-8") as file:
        gold = json.load(file)

    examples = convert_gold(
        gold=gold,
        categories=args.categories,
        include_impossible=args.include_impossible,
    )
    splits = split_examples(
        examples,
        validation_size=args.validation_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    for split_name, rows in splits.items():
        write_jsonl(output_dir / f"{split_name}.jsonl", rows)

    metadata = {
        "source_gold": str(gold_path),
        "num_gold_items": len(gold),
        "num_examples": len(examples),
        "categories": args.categories,
        "include_impossible": args.include_impossible,
        "splits": {
            split_name: {
                "count": len(rows),
                "category_counts": category_counts(rows),
            }
            for split_name, rows in splits.items()
        },
        "load_with": (
            "from datasets import load_dataset; "
            f"ds = load_dataset('json', data_files={{"
            f"'train': '{output_dir / 'train.jsonl'}', "
            f"'validation': '{output_dir / 'validation.jsonl'}', "
            f"'test': '{output_dir / 'test.jsonl'}'}})"
        ),
    }
    write_json(output_dir / "dataset_info.json", metadata)

    print(f"Exemplos QA gerados: {len(examples)}")
    for split_name, rows in splits.items():
        print(f"{split_name}: {len(rows)} {category_counts(rows)}")
    print(f"Guardado em: {output_dir}")


if __name__ == "__main__":
    main()
