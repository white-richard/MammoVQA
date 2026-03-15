"""
Extract LLM decoder features from LLaVA-Mammo for downstream use.

Analogous to extract_features.py (MammoCLIP), but uses LLaVA-Mammo — a fine-tuned
vision-language model — where the meaningful features for mammogram classification
come from the LLM decoder's hidden states (not the vision encoder).

For each image the model receives both the image and a MammoVQA question as a prompt.
The feature vector is the last-layer, last-token hidden state of the Vicuna-7B language
model decoder (shape: D=4096).

Two label modes are supported (--label-mode):

  abnormality (default)
      Uses the MammoVQA "Abnormality" question (multiple-choice).
      label = 0 if answer == ["Normal"], else 1.

  birads
      Uses the MammoVQA "Bi-Rads" question (single-choice).
      label = 1 if Bi-Rads score >= 4 (i.e. Bi-Rads 4, 5, or 6), else 0.

Image paths are sourced from the MammoVQA JSON benchmark files (Train / Eval / Bench
splits) filtered to VinDr-Mammo entries for the selected question topic.  Actual image
files are resolved via the VinDr CSV (patient_id lookup), since the JSON paths reference
a preprocessed layout that may not exist locally.

Features are saved as a .pt file containing:
    {
        "features":    FloatTensor (N, D),
        "labels":      LongTensor  (N,),   # binary: 0=Normal/low-risk, 1=Abnormal/high-risk
        "img_paths":   list[str],
        "metadata":    pd.DataFrame (N, C) -- all JSON entry fields,
        "feature_dim": int,
        "dataset":     str,
        "arch":        str,
        "split":       str,
        "label_col":   "abnormal" | "birads_ge4" | "cancer",
        "label_mode":  "abnormality" | "birads" | "cancer",
    }

Usage
-----
# Unzip model weights first:
#   unzip .../llava-1.6-vicuna-7b_lora-True_qlora-False.zip -d .../Llava-MammoVQA/

# Default (abnormality-based labels):
python extract_features_llava.py \
  --data-dir "$HOME/.code/datasets/vindr-mammo" \
  --img-dir "images_png" \
  --csv-file "vindr_detection_v1_folds_abnormal.csv" \
  --json-dir "$HOME/.code/mammovqa/Benchmark/MammoVQA_JSON" \
  --model-dir "$HOME/.code/model_weights/Llava-MammoVQA/llava-1.6-vicuna-7b_lora-True_qlora-False" \
  --split "all" \
  --dataset "both" \
  --output-file "features/vindr_llava_features.pt" \
  --batch-size 1 \
  --num-workers 0

# Bi-Rads >= 4 labels:
python extract_features_llava.py \
  --data-dir "$HOME/.code/datasets/vindr-mammo" \
  --img-dir "images_png" \
  --csv-file "vindr_detection_v1_folds_abnormal.csv" \
  --json-dir "$HOME/.code/mammovqa/Benchmark/MammoVQA_JSON" \
  --model-dir "$HOME/.code/model_weights/Llava-MammoVQA/llava-1.6-vicuna-7b_lora-True_qlora-False" \
  --split "all" \
  --dataset "both" \
  --label-mode birads \
  --output-file "features/vindr_llava_features_birads.pt" \
  --batch-size 1 \
  --num-workers 0
  
# RSNA:
python extract_features_llava.py \
    --data-dir "$HOME/.code/datasets/rsna/mammo_clip" \
    --img-dir "train_images_png" \
    --csv-file "train_folds.csv" \
    --json-dir "$HOME/.code/mammovqa/Benchmark/MammoVQA_JSON" \
    --model-dir "$HOME/.code/model_weights/Llava-MammoVQA/llava-1.6-vicuna-7b_lora-True_qlora-False" \
    --split "all" \
    --dataset RSNA \
    --label-mode cancer \
    --output-file "features/rsna_llavaVQA_cancer_features.pt" \
    --batch-size 1
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor


def seed_all(seed: int) -> None:
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SINGLE_CHOICE_PREFIX = (
    "This is a mammography-related medical question with several options, only one of which is "
    "correct. Select the correct answer and respond with just the chosen option, without any "
    "further explanation. ### Question: {Question} ### Options: {Options}. ### Answer:"
)

_MULTI_CHOICE_PREFIX = (
    "This is a mammography-related medical question with several options, one or more of which "
    "may be correct. Select the correct answers and respond with only the chosen options, without "
    "any further explanation. ### Question: {Question} ### Options: {Options}. ### Answer:"
)

_ABNORMALITY_OPTIONS = [
    "Normal",
    "Calcification",
    "Mass",
    "Architectural distortion",
    "Asymmetry",
    "Miscellaneous",
    "Nipple retraction",
    "Suspicious lymph node",
    "Skin thickening",
    "Skin retraction",
]

_BIRADS_OPTIONS = [
    "Bi-Rads 0",
    "Bi-Rads 1",
    "Bi-Rads 2",
    "Bi-Rads 3",
    "Bi-Rads 4",
    "Bi-Rads 5",
    "Bi-Rads 6",
]


def build_abnormality_prompt(entry: dict) -> str:
    """Build the shuffled multiple-choice prompt for an Abnormality entry."""
    options = list(entry.get("Options", _ABNORMALITY_OPTIONS))
    random.shuffle(options)
    formatted = ", ".join(f"{chr(65 + i)}: {opt}" for i, opt in enumerate(options))
    return _MULTI_CHOICE_PREFIX.format(Question=entry["Question"], Options=formatted)


def build_birads_prompt(entry: dict) -> str:
    """Build the shuffled single-choice prompt for a Bi-Rads entry."""
    options = list(entry.get("Options", _BIRADS_OPTIONS))
    random.shuffle(options)
    formatted = ", ".join(f"{chr(65 + i)}: {opt}" for i, opt in enumerate(options))
    return _SINGLE_CHOICE_PREFIX.format(Question=entry["Question"], Options=formatted)


_CANCER_QUESTION = "What abnormalities, if any, are present in this mammogram?"


def build_cancer_prompt() -> str:
    """Build a shuffled multiple-choice abnormality prompt for cancer screening (RSNA)."""
    options = list(_ABNORMALITY_OPTIONS)
    random.shuffle(options)
    formatted = ", ".join(f"{chr(65 + i)}: {opt}" for i, opt in enumerate(options))
    return _MULTI_CHOICE_PREFIX.format(Question=_CANCER_QUESTION, Options=formatted)


def birads_answer_to_binary(answer) -> int:
    """
    Convert a Bi-Rads answer to a binary label.

    label = 1 (abnormal/high-risk) if Bi-Rads >= 4, else 0.

    The answer may be a string ("Bi-Rads 4") or a single-element list (["Bi-Rads 4"]).
    The Bi-Rads number is extracted by taking the last whitespace-separated token.
    """
    if isinstance(answer, list):
        answer = answer[0] if answer else ""
    # Extract the numeric part: "Bi-Rads 4" -> "4"
    token = str(answer).strip().split()[-1] if answer else "0"
    try:
        score = int(token)
    except ValueError:
        return 0
    return 1 if score >= 4 else 0


# ---------------------------------------------------------------------------
# JSON loading and filtering
# ---------------------------------------------------------------------------

_SPLIT_FILES = {
    "train": "MammoVQA-Image-Train.json",
    "eval": "MammoVQA-Image-Eval.json",
    "bench": "MammoVQA-Image-Bench.json",
}


def load_json_entries(
    json_dir: Path, split: str, datasets: list[str], question_topic: str = "Abnormality"
) -> list[dict]:
    """
    Load MammoVQA JSON entries filtered by question topic and dataset.

    split: "train" | "eval" | "bench" | "all"
    datasets: list of Dataset values to keep, e.g. ["VinDr-Mammo-breast", "VinDr-Mammo-finding"]
    question_topic: "Abnormality" (default) or "Bi-Rads"
    """
    if split == "all":
        splits_to_load = list(_SPLIT_FILES.keys())
    else:
        if split not in _SPLIT_FILES:
            raise ValueError(f"split must be one of {list(_SPLIT_FILES)} or 'all', got '{split}'")
        splits_to_load = [split]

    entries = []
    for s in splits_to_load:
        json_path = json_dir / _SPLIT_FILES[s]
        print(f"Loading {json_path} ...")
        with open(json_path) as f:
            data = json.load(f)
        for v in data.values():
            if v.get("Question topic") == question_topic and v.get("Dataset") in datasets:
                v["_split"] = s
                entries.append(v)
        print(
            f"  {s}: {sum(1 for e in entries if e.get('_split') == s)} VinDr {question_topic} entries"
        )

    print(f"Total entries: {len(entries)}")
    return entries


# ---------------------------------------------------------------------------
# CSV path index
# ---------------------------------------------------------------------------


def build_path_index(csv_path: Path) -> tuple[dict, dict]:
    """
    Build lookup dicts from the VinDr CSV so we can resolve JSON paths to actual images.

    Returns:
        image_idx:  {image_id_no_ext: (patient_id, image_id_with_ext)}
        series_idx: {series_id: [(patient_id, image_id_with_ext), ...]}
    """
    df = pd.read_csv(csv_path).fillna("")
    image_idx: dict[str, tuple[str, str]] = {}
    series_idx: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for row in df.itertuples(index=False):
        patient_id = str(row.patient_id)
        image_id = str(row.image_id)
        series_id = str(row.series_id)
        image_id_png = image_id if image_id.endswith(".png") else image_id + ".png"
        key = image_id_png.replace(".png", "")
        image_idx[key] = (patient_id, image_id_png)
        series_idx[series_id].append((patient_id, image_id_png))

    print(f"CSV index: {len(image_idx)} images, {len(series_idx)} series")
    return image_idx, series_idx


def resolve_image_path(
    json_path: str,
    image_idx: dict,
    series_idx: dict,
    data_dir: Path,
    img_dir: str,
):
    """
    Map a MammoVQA JSON path to the actual image file.

    VinDr-Mammo-breast:  .../VinDr-Mammo-breast/<image_id_no_ext>/img.jpg
    VinDr-Mammo-finding: .../VinDr-Mammo-finding/<image_id_no_ext>_<k>[_normal]/img.jpg

    In both cases, the first 32 hex chars of the directory name are the image_id.
    """
    parts = json_path.rstrip("/").split("/")
    # parts[-1] = "img.jpg", parts[-2] = dir_name
    dir_name = parts[-2]
    # image_id is the first 32 hex characters of dir_name
    image_id_key = dir_name[:32]
    base = data_dir / img_dir

    entry = image_idx.get(image_id_key)
    if entry is None:
        return None
    patient_id, image_id = entry
    return base / patient_id / image_id


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class LLaVAMammoDataset(Dataset):
    """Dataset that resolves VinDr-Mammo JSON entries to actual images + prompts."""

    def __init__(
        self,
        entries: list[dict],
        image_idx: dict,
        series_idx: dict,
        data_dir: Path,
        img_dir: str,
        label_mode: str = "abnormality",
    ):
        """
        label_mode:
          "abnormality" — binary label from the Abnormality answer
                          (0 = Normal, 1 = any abnormality present)
          "birads"      — binary label from the Bi-Rads answer
                          (0 = Bi-Rads 0-3, 1 = Bi-Rads >= 4)
        """
        if label_mode not in ("abnormality", "birads"):
            raise ValueError(f"label_mode must be 'abnormality' or 'birads', got '{label_mode}'")
        self.label_mode = label_mode

        # Only keep entries where the image file can be resolved
        self.samples = []
        skipped = 0
        for entry in entries:
            img_path = resolve_image_path(entry["Path"], image_idx, series_idx, data_dir, img_dir)
            if img_path is None or not img_path.exists():
                skipped += 1
                continue
            answer = entry.get("Answer", [])
            if label_mode == "birads":
                binary_label = birads_answer_to_binary(answer)
            else:
                binary_label = 0 if answer == ["Normal"] else 1
            self.samples.append((str(img_path), entry, binary_label))

        if skipped:
            print(f"[warning] Skipped {skipped} entries (image not found on disk)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, entry, label = self.samples[idx]
        if self.label_mode == "birads":
            prompt = build_birads_prompt(entry)
        else:
            prompt = build_abnormality_prompt(entry)
        return img_path, prompt, label, entry


def load_rsna_dataframe(data_dir: Path, csv_file: str, split: str) -> pd.DataFrame:
    """
    Load and filter the RSNA CSV for a given split.

    split: "all" | "train" | "test"
        RSNA uses a numeric fold column: fold 0 = validation/test, 1/2 = training.
    """
    df = pd.read_csv(data_dir / csv_file).fillna(0)
    print(f"Full RSNA dataframe shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    if split == "all":
        return df

    fold_col = "fold"
    if fold_col not in df.columns:
        raise ValueError(f"Expected column '{fold_col}' not found in RSNA CSV.")

    if split == "train":
        df = df[df[fold_col].isin([1, 2])].reset_index(drop=True)
    elif split == "test":
        df = df[df[fold_col] == 0].reset_index(drop=True)
    else:
        raise ValueError(f"RSNA split must be 'all', 'train', or 'test', got '{split}'")

    print(f"Filtered RSNA dataframe shape ({split}): {df.shape}")
    return df


class RSNALLaVADataset(Dataset):
    """
    Dataset for RSNA mammography feature extraction via LLaVA-Mammo.

    Reads directly from the RSNA CSV (no MammoVQA JSON required).
    Image paths are resolved as: <data_dir>/<img_dir>/<patient_id>/<image_id>.png
    Labels come from the CSV 'cancer' column (binary 0/1).
    """

    def __init__(self, df: pd.DataFrame, data_dir: Path, img_dir: str):
        self.samples = []
        skipped = 0

        for _, row in df.iterrows():
            patient_id = str(row["patient_id"])
            image_id = str(row["image_id"])
            if not image_id.endswith(".png"):
                image_id = image_id + ".png"
            img_path = data_dir / img_dir / patient_id / image_id
            if not img_path.exists():
                skipped += 1
                continue
            label = int(row.get("cancer", 0))
            self.samples.append((str(img_path), label, row.to_dict()))

        if skipped:
            print(f"[warning] Skipped {skipped} entries (image not found on disk)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, row_dict = self.samples[idx]
        prompt = build_cancer_prompt()
        return img_path, prompt, label, row_dict


def _collate_fn(batch):
    img_paths = [b[0] for b in batch]
    prompts = [b[1] for b in batch]
    labels = [b[2] for b in batch]
    entries = [b[3] for b in batch]
    return img_paths, prompts, labels, entries


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    model_dir: str, device: torch.device
) -> tuple[LlavaNextProcessor, LlavaNextForConditionalGeneration]:
    """
    Load LLaVA-Mammo.

    The model_dir is a PEFT LoRA adapter directory (contains adapter_config.json).
    The base model id is read from adapter_config.json and loaded from HF hub / cache,
    then the LoRA weights are applied and merged for efficient inference.
    """
    adapter_cfg_path = Path(model_dir) / "adapter_config.json"
    if adapter_cfg_path.exists():
        with open(adapter_cfg_path) as f:
            base_model_id = json.load(f)["base_model_name_or_path"]
    else:
        base_model_id = "llava-hf/llava-v1.6-vicuna-7b-hf"

    print(f"Loading processor from {base_model_id} ...")
    processor = LlavaNextProcessor.from_pretrained(base_model_id)

    print(f"Loading base model from {base_model_id} ...")
    base_model = LlavaNextForConditionalGeneration.from_pretrained(
        base_model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )

    print(f"Applying LoRA adapter from {model_dir} ...")
    model = PeftModel.from_pretrained(base_model, model_dir)
    model = model.merge_and_unload()  # merge adapter weights; removes PEFT overhead

    model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Model ready ({num_params:.1f}B parameters)")
    return processor, model


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extract_llava_features(
    model: LlavaNextForConditionalGeneration,
    processor: LlavaNextProcessor,
    loader: DataLoader,
    device: torch.device,
    layer_idx: int = -1,
    debug_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[str], pd.DataFrame]:
    all_features: list[torch.Tensor] = []
    all_labels: list[int] = []
    all_paths: list[str] = []
    all_meta: dict[str, list] = {}

    for i, (img_paths, prompts, labels, entries) in enumerate(tqdm(
        loader, desc="Extracting features", unit="batch"
    )):
        # batch_size=1 enforced; unpack first element
        img_path = img_paths[0]
        prompt = prompts[0]
        label = labels[0]
        entry = entries[0]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[warning] Failed to open {img_path}: {e}; skipping.")
            continue

        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, {"type": "image"}],
            }
        ]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=[image], text=text, return_tensors="pt").to(device)

        # Forward pass — request all hidden states from the LLM decoder
        outputs = model(**inputs, output_hidden_states=True)

        # outputs.hidden_states: tuple of (num_layers+1) tensors, each (B, seq_len, D)
        # Last-layer, last-token → (1, D)
        features = outputs.hidden_states[layer_idx][:, -1, :].float().cpu()

        all_features.append(features)
        all_labels.append(int(label))
        all_paths.append(img_path)

        # Accumulate all JSON entry fields as metadata
        for k, v in entry.items():
            all_meta.setdefault(k, []).append(v)

        if debug_mode and i >= 2:
            print("[debug_mode] Stopping early after 3 batches.")
            break

    metadata_df = pd.DataFrame(all_meta)
    return (
        torch.cat(all_features, dim=0),
        torch.tensor(all_labels, dtype=torch.long),
        all_paths,
        metadata_df,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def config():
    parser = argparse.ArgumentParser(
        description="Extract LLaVA-Mammo LLM decoder features and save them to disk."
    )
    parser.add_argument(
        "--data-dir", required=True, type=str, help="Root directory of the VinDr-Mammo dataset"
    )
    parser.add_argument(
        "--img-dir",
        default="images_png",
        type=str,
        help="Sub-directory containing images (relative to --data-dir)",
    )
    parser.add_argument(
        "--csv-file",
        default="vindr_detection_v1_folds_abnormal.csv",
        type=str,
        help="VinDr CSV file for patient_id/series_id lookup (relative to --data-dir)",
    )
    parser.add_argument(
        "--json-dir",
        required=True,
        type=str,
        help="Directory containing MammoVQA-Image-{Train,Eval,Bench}.json files",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        type=str,
        help="Path to unzipped LLaVA-Mammo checkpoint directory",
    )
    parser.add_argument(
        "--split",
        default="all",
        type=str,
        choices=["all", "train", "eval", "bench", "test"],
        help=(
            "Which split(s) to extract features for. "
            "VinDr: 'train'/'eval'/'bench'/'all'. "
            "RSNA: 'train' (folds 1-2), 'test' (fold 0), or 'all'."
        ),
    )
    parser.add_argument(
        "--dataset",
        default="both",
        type=str,
        choices=["VinDr-Mammo-breast", "VinDr-Mammo-finding", "both", "RSNA"],
        help="Dataset to extract features for. Use 'RSNA' to bypass MammoVQA JSON and read directly from the CSV.",
    )
    parser.add_argument(
        "--layer-idx",
        default=-1,
        type=int,
        help="LLM decoder layer index to extract hidden states from (-1 = last layer)",
    )
    parser.add_argument(
        "--output-file",
        default="features/vindr_llava_features.pt",
        type=str,
        help="Output .pt file path",
    )
    parser.add_argument(
        "--batch-size",
        default=1,
        type=int,
        help="Batch size (must be 1 for LLaVA variable-length inputs)",
    )
    parser.add_argument(
        "--label-mode",
        default="abnormality",
        type=str,
        choices=["abnormality", "birads", "cancer"],
        help=(
            "How to assign binary labels. "
            "'abnormality': uses the Abnormality question; label=1 if any abnormality is present. "
            "'birads': uses the Bi-Rads question; label=1 if Bi-Rads score >= 4."
        ),
    )
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--device", default="cuda", type=str, help="cuda | cpu")
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Cap the number of samples (useful for smoke-testing)",
    )
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Stop after 3 batches — useful for quick end-to-end testing",
    )
    return parser.parse_args()


def main():
    args = config()
    seed_all(args.seed)

    if args.label_mode == "cancer" and args.dataset != "RSNA":
        raise ValueError(
            "--label-mode cancer is only valid with --dataset RSNA. "
            "Did you forget to add --dataset RSNA?"
        )

    if args.batch_size != 1:
        print("[warning] LLaVA requires variable-length processing; forcing batch-size=1")
        args.batch_size = 1

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    print(f"Using device: {device}")

    # ---- Build dataset --------------------------------------------------
    if args.dataset == "RSNA":
        # RSNA: read directly from CSV, no MammoVQA JSON needed
        df = load_rsna_dataframe(
            data_dir=Path(args.data_dir),
            csv_file=args.csv_file,
            split=args.split,
        )
        dataset = RSNALLaVADataset(
            df=df,
            data_dir=Path(args.data_dir),
            img_dir=args.img_dir,
        )
    else:
        # VinDr: load from MammoVQA JSON entries
        if args.dataset == "both":
            datasets = ["VinDr-Mammo-breast", "VinDr-Mammo-finding"]
        else:
            datasets = [args.dataset]

        question_topic = "Bi-Rads" if args.label_mode == "birads" else "Abnormality"
        entries = load_json_entries(
            json_dir=Path(args.json_dir),
            split=args.split,
            datasets=datasets,
            question_topic=question_topic,
        )

        csv_path = Path(args.data_dir) / args.csv_file
        image_idx, series_idx = build_path_index(csv_path)

        dataset = LLaVAMammoDataset(
            entries=entries,
            image_idx=image_idx,
            series_idx=series_idx,
            data_dir=Path(args.data_dir),
            img_dir=args.img_dir,
            label_mode=args.label_mode,
        )

    if args.limit is not None:
        dataset.samples = dataset.samples[: args.limit]
    print(f"Resolved samples: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=_collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    # ---- Load model -----------------------------------------------------
    processor, model = load_model(args.model_dir, device)

    # ---- Extract --------------------------------------------------------
    features, labels, img_paths, metadata_df = extract_llava_features(
        model=model,
        processor=processor,
        loader=loader,
        device=device,
        layer_idx=args.layer_idx,
        debug_mode=args.debug_mode,
    )

    print("\nExtraction complete.")
    print(f"  features shape  : {features.shape}")
    print(f"  labels shape    : {labels.shape}")
    print(f"  unique labels   : {labels.unique().tolist()}")
    print(f"  metadata columns: {list(metadata_df.columns)}")

    # ---- Save -----------------------------------------------------------
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.label_mode == "birads":
        label_col = "birads_ge4"
    elif args.label_mode == "cancer":
        label_col = "cancer"
    else:
        label_col = "abnormal"
    payload = {
        "features": features,  # (N, D) float32
        # "labels": labels,  # (N,)   int64
        "img_paths": img_paths,  # list[str]
        "metadata": metadata_df,  # pd.DataFrame, all JSON entry fields, length N
        "feature_dim": features.shape[1],
        "dataset": args.dataset,
        "arch": "llava-mammo",
        "split": args.split,
        "label_col": label_col,
        "label_mode": args.label_mode,
    }
    torch.save(payload, output_path)
    print(f"\nSaved features to: {output_path}")
    print(f"  feature_dim : {features.shape[1]}")
    print(f"  N samples   : {features.shape[0]}")


if __name__ == "__main__":
    main()
