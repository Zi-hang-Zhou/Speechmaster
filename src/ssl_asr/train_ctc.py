from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import torch
from datasets import Audio, load_dataset
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)

from .metrics import compute_asr_metrics
from .text import normalize_librispeech_text


@dataclass
class DataCollatorCTCWithPadding:
    processor: Any

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        batch = self.processor.pad(input_features, padding=True, return_tensors="pt")
        labels_batch = self.processor.pad(labels=label_features, padding=True, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def prepare_dataset(batch: dict, processor) -> dict:
    audio = batch["audio"]
    batch["input_values"] = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    ).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(normalize_librispeech_text(batch["text"])).input_ids
    return batch


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)

    processor = AutoProcessor.from_pretrained(args.processor or args.model)
    ds = load_dataset(
        "openslr/librispeech_asr",
        args.subset,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )
    train = ds[args.train_split].select(range(min(args.train_limit, len(ds[args.train_split]))))
    eval_ds = ds[args.eval_split].select(range(min(args.eval_limit, len(ds[args.eval_split]))))
    train = train.cast_column("audio", Audio(sampling_rate=16000))
    eval_ds = eval_ds.cast_column("audio", Audio(sampling_rate=16000))
    train = train.map(lambda b: prepare_dataset(b, processor), remove_columns=train.column_names)
    eval_ds = eval_ds.map(lambda b: prepare_dataset(b, processor), remove_columns=eval_ds.column_names)

    model = AutoModelForCTC.from_pretrained(
        args.model,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ignore_mismatched_sizes=True,
    )
    if args.freeze_feature_encoder and hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    if args.freeze_encoder:
        for name, param in model.named_parameters():
            if "lm_head" not in name:
                param.requires_grad = False

    def compute_metrics(pred):
        pred_ids = torch.tensor(pred.predictions).argmax(dim=-1).numpy()
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        metrics = compute_asr_metrics(label_str, pred_str)
        return {"wer": metrics.wer, "cer": metrics.cer}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        fp16=args.fp16 and torch.cuda.is_available(),
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.num_workers,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        save_total_limit=2,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=eval_ds,
        tokenizer=processor,
        data_collator=DataCollatorCTCWithPadding(processor),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(trainer.evaluate())


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune a low-resource SSL CTC ASR model.")
    p.add_argument("--model", default="facebook/wav2vec2-base")
    p.add_argument("--processor", default="facebook/wav2vec2-base-960h")
    p.add_argument("--subset", default="clean")
    p.add_argument("--train-split", default="train.100")
    p.add_argument("--eval-split", default="validation")
    p.add_argument("--train-limit", type=int, default=1000)
    p.add_argument("--eval-limit", type=int, default=256)
    p.add_argument("--cache-dir", default="data/hf_cache")
    p.add_argument("--output-dir", default="results/checkpoints/wav2vec2_low_resource")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--logging-steps", type=int, default=25)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--freeze-feature-encoder", action="store_true")
    p.add_argument("--freeze-encoder", action="store_true")
    return p


if __name__ == "__main__":
    main()

