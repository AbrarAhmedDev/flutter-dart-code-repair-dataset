# Flutter & Dart Code Repair Dataset

**A supervised fine-tuning (SFT) dataset for Flutter/Dart error diagnosis and full-file code repair.**

Fine-tune LLMs that actually *fix* Flutter and Dart code — not just generate it.

[![Format](https://img.shields.io/badge/format-JSONL%20%2F%20ChatML-0468D7)]()
[![Verified](https://img.shields.io/badge/verified-0%20analyzer%20errors-3ED98B)]()
[![Compatible](https://img.shields.io/badge/compatible-Unsloth%20%7C%20TRL%20%7C%20LLaMA--Factory%20%7C%20Axolotl-27C6FF)]()

**[→ Get the full dataset (32,323 records + Unsloth training script)](https://5220745837352.gumroad.com/l/flutter-code-repair-dataset)**

---

## Why this exists

General-purpose code LLMs are noticeably weaker at Flutter and Dart than at more heavily-represented languages, and weaker still at *repairing* Flutter/Dart code than at writing it from scratch. The widget tree, the `State`/lifecycle model, and Dart's null-safety rules don't behave like generic OOP code — so a model that's good at Python bug-fixing doesn't automatically transfer.

This repo documents a dataset built to close that specific gap: verified before/after pairs of real, full-file Flutter/Dart errors and their corrected fixes, formatted for supervised fine-tuning.

This repository contains the **documentation, schema, verification methodology, a free sample, and a training quickstart.** The full 32,323-record dataset and 1-click Unsloth training script are distributed separately (see [Full dataset & pricing](#full-dataset--pricing)) — this keeps the repo genuinely useful on its own while the paid product funds the ongoing curation and verification work.

## What makes this dataset different

- **Full-file context.** Examples are 150–250 line real-world Flutter screens, not isolated snippets — closer to what a model actually has to reason about in production.
- **Multiple simultaneous errors.** Each file contains 2–10 errors distributed across it, matching how real bugs cluster rather than appearing in isolation.
- **Verified corrections.** Every corrected sample passes Dart static analysis with 0 analyzer errors before being included — see [Verification methodology](#verification-methodology).
- **Four prompt formats**, so a model learns to work from whatever signal is actually available at inference time — a compiler error, IDE feedback, raw code with no context, or a request for root-cause reasoning.

## Error taxonomy

| Category | Examples |
|---|---|
| Layout & hierarchy | `Expanded` used outside a `Flex`, unconstrained `ListView`, incorrect `Positioned` placement |
| Lifecycle & state safety | Missing `mounted` checks after `await`, controller memory leaks, `FutureBuilder` rebuild bugs |
| Constructor & type safety | Required parameter mismatches, named vs. positional argument errors |
| Null safety | Null-safety violations across 40+ widgets (`Container`, `TextField`, buttons, `TabBar`, cards, navigation, and more) |

## Prompt formats (ChatML)

| Format | Description |
|---|---|
| `compiler_diagnostics` | Model is given the raw Dart analyzer/compiler error output alongside the broken file |
| `raw_code_only` | Model is given only the broken file, with no external error signal — closest to "spot the bug" |
| `ide_feedback` | Model is given IDE-style inline diagnostics (as a developer would see them in an editor) |
| `root_cause_analysis` | Chain-of-thought format requiring the model to explain *why* the error occurs before producing the fix |

## Dataset schema

Every record follows a ChatML-style structure. The example below illustrates the **shape only** — it is not a verbatim dataset entry.

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a Flutter/Dart code repair assistant. Diagnose and fix the reported error(s) in the provided file."
    },
    {
      "role": "user",
      "content": "// compiler_diagnostics format\n// error • lib/screens/profile_screen.dart:142:9 • The named parameter 'onTap' is required, but there's no corresponding argument\n\n<full 150–250 line Flutter file omitted here for brevity>"
    },
    {
      "role": "assistant",
      "content": "<corrected full file, verified against dart analyze with 0 errors>"
    }
  ],
  "metadata": {
    "prompt_format": "compiler_diagnostics",
    "error_categories": ["constructor_type_safety"],
    "error_count": 3,
    "widget_count": 12,
    "line_count": 187
  }
}
```

## Verification methodology

Every corrected file in this dataset is run through `dart analyze` after the fix is applied. A sample is only included if the analyzer reports **0 errors**. This doesn't guarantee the fix is the only valid one, or that it's stylistically ideal — it guarantees the corrected code is at minimum syntactically and statically valid Dart, which rules out an entire class of hallucinated "fixes" that look plausible but don't actually compile.

## Quickstart: fine-tuning with Unsloth

The full dataset (Pro tier) ships with a ready-to-run Unsloth QLoRA script for Google Colab or Kaggle. At a high level, the workflow looks like this:

```python
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# 1. Load a base model in 4-bit for QLoRA fine-tuning
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",
    max_seq_length=4096,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=16)

# 2. Load the Flutter/Dart code repair dataset (JSONL / ChatML)
dataset = load_dataset("json", data_files="flutter_dart_repair_train.jsonl", split="train")

# 3. Fine-tune with TRL's SFTTrainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(per_device_train_batch_size=2, num_train_epochs=3, output_dir="outputs"),
)
trainer.train()
```

> The exact, tested notebook (with the correct hyperparameters for this dataset) is included with the Pro tier purchase — the snippet above shows the shape of the workflow, not the shipped script.

## Sample data

A free sample of records is included in [`/sample`](./sample) so you can inspect the format and run it through your own pipeline before buying. It's a genuine subset of the full dataset's structure and quality — not a cut-down teaser.

## Full dataset & pricing

| Tier | Price | Contents |
|---|---|---|
| **Starter Pack** | $5 | 2,000 balanced records, JSONL/ChatML |
| **Pro Grand Master** | $40 | All 32,323 records, structured JSON, + 1-click Unsloth QLoRA training script |

**[→ Get the dataset on Gumroad](https://5220745837352.gumroad.com/l/flutter-code-repair-dataset)**

## Compatibility

JSONL / ChatML format, compatible with:

- [Unsloth](https://github.com/unslothai/unsloth)
- [Hugging Face TRL](https://github.com/huggingface/trl)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)

## FAQ

**What is a Flutter code repair dataset?**
A collection of labeled Flutter/Dart code examples pairing broken code with its verified, corrected version, used to train or fine-tune a model to diagnose and fix real errors rather than just generate new code.

**Can I use this to fine-tune an open-source model?**
Yes — that's the primary use case. The format and included training script are built specifically for SFT/QLoRA fine-tuning workflows.

**Is the corrected code guaranteed to be bug-free?**
Every corrected sample passes `dart analyze` with 0 errors, meaning it's statically valid Dart. This confirms static correctness, not runtime behavior or code style.

**Do I need a powerful GPU?**
The included QLoRA script targets free/low-cost environments like Google Colab or Kaggle notebooks.

## License & citation

The contents of this repository (documentation, sample data, and code) are provided under the [MIT License](./LICENSE). The full paid dataset has its own usage terms, available on the [Gumroad listing](https://5220745837352.gumroad.com/l/flutter-code-repair-dataset).

If you use this dataset in research, please cite:

```
@misc{flutter_dart_code_repair_dataset,
  title  = {Flutter \& Dart Code Repair Dataset},
  author = {AbrarDev},
  year   = {2026},
  note   = {Available at https://5220745837352.gumroad.com/l/flutter-code-repair-dataset}
}
```

## Issues & contributions

Questions about the schema, verification methodology, or the sample data are welcome via [Issues](../../issues). This repo is actively maintained alongside the dataset.
