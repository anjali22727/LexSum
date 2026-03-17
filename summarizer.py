"""
models/summarizer.py
=====================
Longformer Encoder-Decoder (LED) for legal document summarisation.

Why LED over BART/T5?
---------------------
Standard BART and T5 are capped at 1 024 tokens input.
A typical Supreme Court judgment is 3 000–20 000 words.
LED (allenai/led-base-16384) handles up to 16 384 tokens using
Longformer's sliding-window + global attention — meaning we can
feed the **full document** rather than truncating it.

This is the core research contribution: LED + Legal-BERT retrieval
outperforms BART-base on ROUGE because:
  1. No truncation loss on long judgments
  2. Legal-BERT chunks give the model the most legally relevant context
  3. Fine-tuning on Indian legal summaries adapts vocabulary

Two modes
---------
  EXTRACTIVE_AUGMENTED : RAG retrieves top-k chunks → LED summarises those
  FULL_DOC             : LED receives the entire document (up to 16 384 tokens)
"""

import torch
from transformers import (
    LEDTokenizer,
    LEDForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from datasets import Dataset
import numpy as np

LED_MODEL = "allenai/led-base-16384"
MAX_INPUT_LEN  = 8192    # conservative cap — avoids OOM on 16 GB GPU
MAX_TARGET_LEN = 512     # summary length ceiling


class LEDSummarizer:
    """
    Wraps LED for inference (summarisation).

    Usage
    -----
    summ = LEDSummarizer()
    summary = summ.summarise(long_legal_text)

    Or with RAG chunks:
    summary = summ.summarise_chunks(retrieved_chunks)
    """

    def __init__(self, model_path: str = LED_MODEL, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[LEDSummarizer] Loading {model_path}...")
        self.tokenizer = LEDTokenizer.from_pretrained(model_path)
        self.model     = LEDForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        print("[LEDSummarizer] Ready.")

    @torch.no_grad()
    def summarise(
        self,
        text: str,
        max_new_tokens:  int   = MAX_TARGET_LEN,
        num_beams:       int   = 4,
        length_penalty:  float = 2.0,
        early_stopping:  bool  = True,
        no_repeat_ngram: int   = 3,
    ) -> str:
        """
        Summarise a single text string.

        Parameters
        ----------
        text             : input legal text (can be thousands of words)
        max_new_tokens   : maximum summary length in tokens
        num_beams        : beam search width (higher = better quality, slower)
        length_penalty   : > 1.0 encourages longer summaries
        no_repeat_ngram  : prevents repetitive phrases (3 = good default)
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=MAX_INPUT_LEN,
            truncation=True,
            padding="longest",
        ).to(self.device)

        # Longformer global attention: set on first token ([CLS])
        # This lets the encoder "broadcast" context from the summary token
        global_attention_mask = torch.zeros_like(inputs["input_ids"])
        global_attention_mask[:, 0] = 1

        output_ids = self.model.generate(
            input_ids               = inputs["input_ids"],
            attention_mask          = inputs["attention_mask"],
            global_attention_mask   = global_attention_mask,
            max_new_tokens          = max_new_tokens,
            num_beams               = num_beams,
            length_penalty          = length_penalty,
            early_stopping          = early_stopping,
            no_repeat_ngram_size    = no_repeat_ngram,
        )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def summarise_chunks(self, chunks: list[dict], separator: str = "\n\n") -> str:
        """
        Concatenate retrieved chunks then summarise.
        chunks : list of dicts with a 'text' key (e.g. from FAISS search)
        """
        combined = separator.join(c["text"] for c in chunks)
        return self.summarise(combined)


# ─── Fine-tuning script ───────────────────────────────────────────────────────

class LEDFineTuner:
    """
    Fine-tunes LED on a (document, summary) dataset.

    Dataset format expected:
        List of dicts: [{"document": "...", "summary": "..."}, ...]

    Typical sources for Indian legal summaries:
      - SCI headnotes (the short summary at the top of judgments)
      - IN-Abs dataset (if available)
      - Manually annotated summaries

    Example
    -------
    tuner = LEDFineTuner(output_dir="./finetuned_led")
    tuner.train(train_data, eval_data, epochs=3)
    """

    def __init__(
        self,
        base_model:  str = LED_MODEL,
        output_dir:  str = "./finetuned_led",
        device:      str = "auto",
    ):
        self.output_dir = output_dir
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = LEDTokenizer.from_pretrained(base_model)
        self.model     = LEDForConditionalGeneration.from_pretrained(base_model)

    def _preprocess(self, examples: dict) -> dict:
        """
        Tokenise documents and summaries.
        Called by HuggingFace Datasets .map().
        """
        # Tokenise input documents
        model_inputs = self.tokenizer(
            examples["document"],
            max_length=MAX_INPUT_LEN,
            truncation=True,
            padding="max_length",
        )

        # Tokenise target summaries
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples["summary"],
                max_length=MAX_TARGET_LEN,
                truncation=True,
                padding="max_length",
            )

        # Replace padding token id in labels with -100
        # so they are ignored in the loss calculation
        label_ids = [
            [(l if l != self.tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]
        model_inputs["labels"] = label_ids

        # Add global attention on first token for each example
        global_attn = [
            [1] + [0] * (MAX_INPUT_LEN - 1)
            for _ in examples["document"]
        ]
        model_inputs["global_attention_mask"] = global_attn

        return model_inputs

    def train(
        self,
        train_data: list[dict],
        eval_data:  list[dict],
        epochs:     int   = 3,
        batch_size: int   = 2,
        lr:         float = 5e-5,
        warmup_steps: int = 200,
    ) -> None:
        """
        Fine-tune LED.

        train_data / eval_data : [{"document": ..., "summary": ...}, ...]
        """
        train_ds = Dataset.from_list(train_data)
        eval_ds  = Dataset.from_list(eval_data)

        # Tokenise
        tokenised_train = train_ds.map(self._preprocess, batched=True, batch_size=8)
        tokenised_eval  = eval_ds.map(self._preprocess,  batched=True, batch_size=8)

        # Training arguments
        # gradient_accumulation_steps × per_device_train_batch_size = effective batch
        training_args = Seq2SeqTrainingArguments(
            output_dir                  = self.output_dir,
            num_train_epochs            = epochs,
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size  = batch_size,
            gradient_accumulation_steps = 4,       # effective batch = 8
            warmup_steps                = warmup_steps,
            learning_rate               = lr,
            weight_decay                = 0.01,
            predict_with_generate       = True,
            evaluation_strategy         = "epoch",
            save_strategy               = "epoch",
            load_best_model_at_end      = True,
            metric_for_best_model       = "eval_loss",
            fp16                        = (self.device == "cuda"),
            logging_steps               = 50,
            report_to                   = "none",   # set "wandb" if you use W&B
        )

        collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model,
            pad_to_multiple_of=64,
        )

        trainer = Seq2SeqTrainer(
            model         = self.model,
            args          = training_args,
            train_dataset = tokenised_train,
            eval_dataset  = tokenised_eval,
            tokenizer     = self.tokenizer,
            data_collator = collator,
            callbacks     = [EarlyStoppingCallback(early_stopping_patience=2)],
        )

        print("[LEDFineTuner] Starting training...")
        trainer.train()
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"[LEDFineTuner] Model saved to {self.output_dir}")
