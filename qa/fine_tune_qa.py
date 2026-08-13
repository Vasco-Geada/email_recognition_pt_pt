import argparse
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset


DEFAULT_MODEL = "pierreguillou/bert-base-cased-squad-v1.1-portuguese"
LOGGER = logging.getLogger("fine_tune_qa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune BERTimbau QA with AutoModelForQuestionAnswering."
    )
    parser.add_argument("--train-file", default="dataset/hf_qa_train_validation/train.jsonl")
    parser.add_argument("--validation-file", default="dataset/hf_qa_train_validation/validation.jsonl")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="qa/models/bertimbau_qa_finetuned")
    parser.add_argument("--cache-dir", default="dataset/.hf_cache")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--doc-stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--num-epochs", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fp16", action="store_true", help="Usa mixed precision em CUDA.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--save-every-epoch", action="store_true")
    return parser.parse_args()


# Set random seeds for reproducibility across random, numpy, and torch.
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Resolve the device to use for training (CPU or CUDA) based on user input and availability.
def resolve_device(requested_device: str) -> torch.device:
    requested = str(requested_device or "auto").lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"

    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "Foi pedido --device cuda, mas este ambiente Python nao tem CUDA ativo. "
            f"PyTorch instalado: {torch.__version__}; torch.version.cuda={torch.version.cuda}. "
            "Instala uma build CUDA do PyTorch na .venv ou corre com --device cpu."
        )

    device = torch.device(requested)
    if device.type == "cuda":
        LOGGER.info("CUDA ativo: %s", torch.cuda.get_device_name(device))
    else:
        LOGGER.info("A usar CPU")
    return device


# Load the QA dataset from JSON files for training and validation, using the Hugging Face datasets library.
def load_qa_dataset(train_file: str, validation_file: str, cache_dir: str):
    return load_dataset(
        "json",
        data_files={
            "train": train_file,
            "validation": validation_file,
        },
        cache_dir=cache_dir,
    )

# Prepare the training features for the QA model by tokenizing the questions and contexts, handling overflow, and mapping answer positions to token indices.
def prepare_train_features(examples: Dict[str, List[Any]], tokenizer, args: argparse.Namespace) -> Dict[str, List[Any]]:
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=args.max_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    start_positions = []
    end_positions = []
    example_ids = []
    categories = []

# Map the answer start and end character positions to token indices, handling cases where the answer is not fully contained in the tokenized span.
    for feature_index, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][feature_index]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized.sequence_ids(feature_index)
        sample_index = sample_mapping[feature_index]
        answers = examples["answers"][sample_index]
        example_ids.append(examples["id"][sample_index])
        categories.append(examples.get("category", [""])[sample_index])

        if not answers["answer_start"]:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        start_char = answers["answer_start"][0]
        answer_text = answers["text"][0]
        end_char = start_char + len(answer_text)

        token_start_index = 0
        while sequence_ids[token_start_index] != 1:
            token_start_index += 1

        token_end_index = len(input_ids) - 1
        while sequence_ids[token_end_index] != 1:
            token_end_index -= 1

        if offsets[token_start_index][0] > start_char or offsets[token_end_index][1] < end_char:
            start_positions.append(cls_index)
            end_positions.append(cls_index)
            continue

        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        start_positions.append(token_start_index - 1)

        while offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        end_positions.append(token_end_index + 1)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    tokenized["example_id"] = example_ids
    tokenized["category"] = categories
    return tokenized

# Collate a batch of examples into tensors for input to the model, including input IDs, attention masks, token type IDs, and answer positions, while also preserving example IDs and categories.
def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensor_keys = ["input_ids", "attention_mask", "start_positions", "end_positions"]
    if "token_type_ids" in batch[0]:
        tensor_keys.append("token_type_ids")

    result = {
        key: torch.tensor([item[key] for item in batch], dtype=torch.long)
        for key in tensor_keys
    }
    result["example_id"] = [item["example_id"] for item in batch]
    result["category"] = [item["category"] for item in batch]
    return result

# Create an optimizer for the model with separate weight decay settings for different parameter groups.
def make_optimizer(model, args: argparse.Namespace):
    no_decay = ["bias", "LayerNorm.weight"]
    grouped_parameters = [
        {
            "params": [
                parameter for name, parameter in model.named_parameters()
                if not any(nd in name for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                parameter for name, parameter in model.named_parameters()
                if any(nd in name for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(grouped_parameters, lr=args.learning_rate)

# Evaluate the average loss of the model on the evaluation dataset, using mixed precision if specified.
def evaluate_loss(model, dataloader: DataLoader, device: torch.device, use_fp16: bool) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key in {"input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"}
            }
            with torch.autocast(device_type="cuda", enabled=use_fp16):
                outputs = model(**inputs)
            losses.append(float(outputs.loss.detach().cpu()))
    model.train()
    return sum(losses) / len(losses) if losses else 0.0

# Save the model and tokenizer to the specified output directory, along with training metrics in a JSON file.
def save_checkpoint(model, tokenizer, output_dir: Path, metrics: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    with (output_dir / "training_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    set_seed(args.seed)
    device = resolve_device(args.device)
    use_fp16 = bool(args.fp16 and device.type == "cuda")
    if args.fp16 and device.type != "cuda":
        LOGGER.warning("--fp16 foi pedido, mas mixed precision so sera usado em CUDA.")

    LOGGER.info("A carregar dataset QA")
    raw_dataset = load_qa_dataset(args.train_file, args.validation_file, args.cache_dir)
    if args.max_train_samples:
        raw_dataset["train"] = raw_dataset["train"].select(range(min(args.max_train_samples, len(raw_dataset["train"]))))
    if args.max_eval_samples:
        raw_dataset["validation"] = raw_dataset["validation"].select(range(min(args.max_eval_samples, len(raw_dataset["validation"]))))

    LOGGER.info("A carregar tokenizer/modelo: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, cache_dir=args.cache_dir)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    model.to(device)

    LOGGER.info("A tokenizar dataset")
    tokenized = raw_dataset.map(
        lambda examples: prepare_train_features(examples, tokenizer, args),
        batched=True,
        remove_columns=raw_dataset["train"].column_names,
    )

    train_loader = DataLoader(
        tokenized["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    
    eval_loader = DataLoader(
        tokenized["validation"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    optimizer = make_optimizer(model, args)
    total_update_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.num_epochs
    LOGGER.info("Treino: %s batches, %s update steps", len(train_loader), total_update_steps)

    metrics = {
        "model_name": args.model_name,
        "train_file": args.train_file,
        "validation_file": args.validation_file,
        "num_train_features": len(tokenized["train"]),
        "num_validation_features": len(tokenized["validation"]),
        "epochs": [],
    }

    model.train()
    global_step = 0
    # Train the model for the specified number of epochs, accumulating gradients and updating the optimizer at defined intervals, while logging training and validation loss metrics.
    for epoch in range(args.num_epochs):
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.num_epochs}")
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(progress, start=1):
            inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key in {"input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"}
            }
            with torch.autocast(device_type="cuda", enabled=use_fp16):
                outputs = model(**inputs)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += float(loss.detach().cpu()) * args.gradient_accumulation_steps

            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            progress.set_postfix(loss=f"{running_loss / step:.4f}")

        train_loss = running_loss / len(train_loader) if train_loader else 0.0
        eval_loss = evaluate_loss(model, eval_loader, device, use_fp16)
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": eval_loss,
            "global_step": global_step,
        }
        metrics["epochs"].append(epoch_metrics)
        LOGGER.info(
            "Epoch %s/%s: train_loss=%.4f validation_loss=%.4f",
            epoch + 1,
            args.num_epochs,
            train_loss,
            eval_loss,
        )

        if args.save_every_epoch:
            save_checkpoint(
                model,
                tokenizer,
                Path(args.output_dir) / f"epoch_{epoch + 1}",
                {**metrics, "latest_epoch": epoch_metrics},
            )

    save_checkpoint(model, tokenizer, Path(args.output_dir), metrics)
    LOGGER.info("Modelo fine-tuned guardado em: %s", Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()
