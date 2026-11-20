import json
import os
import shutil
import subprocess
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import torch
from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from be.config import settings

DRY_RUN = settings.DRY_RUN
DRY_RUN_LIMIT = settings.DRY_RUN_LIMIT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = settings.CLIP_MODEL_ID
DATASET_ID = settings.DATASET_ID

BATCH_SIZE = settings.BATCH_SIZE
NUM_WORKERS = settings.NUM_WORKERS
CHECKPOINT_FREQ = settings.CHECKPOINT_FREQ

RAW_DATA_DIR = settings.RAW_DATA_DIR
BASE_OUTPUT_DIR = settings.CHECKPOINTS_DIR


def extract_caption(record: dict[str, Any]) -> str:
    if record.get("short_caption"):
        return str(record["short_caption"])
    if record.get("text"):
        return str(record["text"])

    conversations = record.get("conversations")
    if isinstance(conversations, list):
        for turn in conversations:
            if turn.get("from") in ("gpt", "assistant") and turn.get("value"):
                return str(turn["value"]).strip()
        for turn in conversations:
            if turn.get("value"):
                return str(turn["value"]).replace("<image>", "").strip()
    return ""


def get_dataset_file(filename: str) -> Path:

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    local_path = RAW_DATA_DIR / filename
    if local_path.exists() and local_path.stat().st_size > 0:
        print(f"[INFO] Używam istniejącego pliku lokalnego: {local_path}")
        return local_path

    print(
        f"[INFO] Plik {filename} nie został znaleziony w {RAW_DATA_DIR}. "
        f"Pobieranie z Hugging Face ({DATASET_ID})..."
    )
    downloaded = hf_hub_download(
        repo_id=DATASET_ID,
        filename=filename,
        repo_type="dataset",
        local_dir=str(RAW_DATA_DIR),
    )
    return Path(downloaded)


def prepare_dry_run_data() -> tuple[list[dict[str, Any]], Path]:
    jsonl_path = get_dataset_file("eval.jsonl")
    extract_dir = RAW_DATA_DIR / "eval_images"

    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        zip_path = get_dataset_file("eval_images.zip")
        print(f"[INFO] Rozpakowywanie {zip_path.name} do {extract_dir}...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    else:
        print(f"[INFO] Używam już rozpakowanego katalogu: {extract_dir}")

    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    unique_records = {}
    for r in records:
        rel_img = r.get("image")
        if rel_img and rel_img not in unique_records:
            unique_records[rel_img] = r

    final_records = list(unique_records.values())
    if DRY_RUN_LIMIT is not None:
        final_records = final_records[:DRY_RUN_LIMIT]

    return final_records, extract_dir


def prepare_full_train_data() -> tuple[list[dict[str, Any]], Path]:

    conv_path = get_dataset_file("conversations.json")
    extract_dir = RAW_DATA_DIR / "train_images"

    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        archive_files = [
            "images.z01",
            "images.z02",
            "images.z03",
            "images.z04",
            "images.z05",
            "images.zip",
        ]
        full_zip = RAW_DATA_DIR / "images_full.zip"
        split_zip = RAW_DATA_DIR / "images.zip"

        if not full_zip.exists() and not all(
            (RAW_DATA_DIR / f).exists() for f in archive_files
        ):
            print("[INFO] Sprawdzanie i pobieranie brakujących części archiwum...")
            for fname in archive_files:
                get_dataset_file(fname)

        zip_to_extract = full_zip if full_zip.exists() else split_zip

        seven_zip_bin = shutil.which("7z") or shutil.which("7za")
        if seven_zip_bin:
            print(
                f"[INFO] Znaleziono 7-Zip ({seven_zip_bin}). "
                f"Rozpakowywanie archiwum {zip_to_extract.name} do {extract_dir}..."
            )
            extract_dir.mkdir(parents=True, exist_ok=True)
            try:
                subprocess.run(
                    [seven_zip_bin, "x", str(zip_to_extract), f"-o{extract_dir}", "-y"],
                    check=True,
                )
                print(f"[SUKCES] Rozpakowano obrazy do {extract_dir}.")
            except subprocess.CalledProcessError as err:
                raise RuntimeError(
                    f"Błąd podczas rozpakowywania 7-Zip: {err}. "
                    f"Spróbuj rozpakować ręcznie poleceniem:\n"
                    f'  7z x "{zip_to_extract}" -o"{extract_dir}" -y'
                ) from err
        else:
            raise RuntimeError(
                f"\n[BLOKADA] Archiwa są obecne w {RAW_DATA_DIR}, ale obrazy nie zostały jeszcze "
                f"rozpakowane do {extract_dir}, a w systemie nie odnaleziono polecenia '7z'.\n"
                f"Zainstaluj 7-Zip (np. 'sudo apt install p7zip-full') lub rozpakuj ręcznie:\n"
                f'  7z x "{zip_to_extract}" -o"{extract_dir}"\n'
                f"Po zakończeniu rozpakowywania zrestartuj ten skrypt.\n"
            )
    else:
        print(f"[INFO] Używam już rozpakowanego katalogu z obrazami: {extract_dir}")

    print(f"[INFO] Wczytywanie i deduplikacja {conv_path.name} (~3.4 GB JSON)...")
    start_parse = time.perf_counter()
    with conv_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    unique_records = {}
    for record in records:
        rel_img = record.get("image")
        if rel_img and isinstance(rel_img, str) and rel_img not in unique_records:
            unique_records[rel_img] = record

    elapsed = time.perf_counter() - start_parse
    print(
        f"[INFO] Sparsowano JSON w {elapsed:.1f} s. "
        f"Liczba kadrów po deduplikacji obrazów: {len(unique_records)} (z początkowych {len(records)})"
    )

    return list(unique_records.values()), extract_dir


def apply_resume_filter(
    records: list[dict[str, Any]], checkpoints_dir: Path
) -> tuple[list[dict[str, Any]], int]:

    if not checkpoints_dir.exists():
        return records, 1

    parquet_files = sorted(checkpoints_dir.glob("features_ckpt_*.parquet"))
    if not parquet_files:
        return records, 1

    print("[RESUME] Wykryto istniejące checkpointy. Analizowanie stanu postępu...")
    processed_images: set[str] = set()
    highest_idx = 0

    for path in parquet_files:
        try:
            df = pd.read_parquet(path, engine="pyarrow", columns=["image_path"])
            processed_images.update(df["image_path"].tolist())

            file_num = int(path.stem.split("_")[-1])
            highest_idx = max(highest_idx, file_num)
        except (pa.ArrowInvalid, OSError, ValueError, KeyError) as e:
            print(f"[WARN] Zignorowano uszkodzony checkpoint '{path.name}': {e}")

    if not processed_images:
        return records, 1

    remaining = [r for r in records if str(r.get("image", "")) not in processed_images]
    print(
        f"[RESUME] Ominięto {len(records) - len(remaining)} przetworzonych rekordów. "
        f"Pozostało: {len(remaining)}"
    )

    return remaining, highest_idx + 1


class VideoGameBunnyDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], images_root: Path) -> None:
        self.records = records
        self.images_root = images_root
        self.suffix_index: dict[str, Path] = {}
        self.basename_index: dict[str, Path] = {}
        self._build_file_index()

        if self.records:
            sample_rel = str(self.records[0].get("image", ""))
            resolved_sample = self._resolve_image_path(sample_rel)
            if resolved_sample is None:
                sample_files = list(self.suffix_index.values())[:5]
                raise FileNotFoundError(
                    f"Nie udało się dopasować ścieżki obrazu '{sample_rel}' w katalogu {images_root}.\n"
                    f"Przykładowe pliki: {sample_files}"
                )

    def _build_file_index(self) -> None:
        print("[INFO] Budowanie szybkiego indeksu ścieżek obrazów...")
        for root, _, files in os.walk(self.images_root):
            root_path = Path(root)
            for fname in files:
                full_path = root_path / fname
                rel_posix = full_path.relative_to(self.images_root).as_posix().lower()
                parts = rel_posix.split("/")
                self.suffix_index[rel_posix] = full_path
                if len(parts) >= 2:
                    tail_two = "/".join(parts[-2:])
                    self.suffix_index.setdefault(tail_two, full_path)
                self.basename_index.setdefault(fname.lower(), full_path)

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image_path(self, rel_path: str) -> Path | None:
        norm = rel_path.replace("\\", "/").lstrip("/").lower()
        if norm in self.suffix_index:
            return self.suffix_index[norm]

        parts = norm.split("/")
        if len(parts) >= 2:
            tail_two = "/".join(parts[-2:])
            if tail_two in self.suffix_index:
                return self.suffix_index[tail_two]

        return self.basename_index.get(parts[-1])

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        record = self.records[idx]
        rel_image = record.get("image")
        if not rel_image or not isinstance(rel_image, str):
            return None

        img_path = self._resolve_image_path(rel_image)
        if img_path is None:
            return None

        try:
            with Image.open(img_path) as img:
                rgb_image = img.convert("RGB")
        except (OSError, UnidentifiedImageError):
            return None

        sample_id_from_id = record.get("id") or str(idx)
        return {
            "sample_idx": sample_id_from_id,
            "image_path": rel_image,
            "category": str(record.get("category") or ""),
            "caption": extract_caption(record),
            "image": rgb_image,
        }


class ClipCollator:
    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.processor: CLIPProcessor | None = None

    def __call__(
        self, batch: list[dict[str, Any] | None]
    ) -> tuple[dict[str, torch.Tensor] | None, list[dict[str, Any]]]:
        if self.processor is None:
            self.processor = CLIPProcessor.from_pretrained(
                self.model_id, local_files_only=True
            )

        valid_items = [item for item in batch if item is not None]
        if not valid_items:
            return None, []

        images = [item["image"] for item in valid_items]
        inputs = self.processor(images=images, return_tensors="pt")

        metadata = [
            {
                "sample_idx": str(item["sample_idx"]),
                "image_path": str(item["image_path"]),
                "category": str(item["category"]),
                "caption": str(item["caption"]),
            }
            for item in valid_items
        ]
        return dict(inputs), metadata


def save_checkpoint(
    features: list[list[float]],
    metadata: list[dict[str, Any]],
    ckpt_idx: int,
    output_dir: Path,
) -> None:
    if not features:
        return

    filename = output_dir / f"features_ckpt_{ckpt_idx:04d}.parquet"
    df = pd.DataFrame(metadata)
    df["embedding"] = features
    df.to_parquet(filename, engine="pyarrow", index=False)


def main() -> None:
    if DEVICE != "cuda":
        raise RuntimeError("CUDA nie jest dostępna!")

    BASE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()

    mode_label = (
        f"DRY RUN (eval, limit={DRY_RUN_LIMIT})"
        if DRY_RUN
        else "PEŁNY PRZEBIEG (train)"
    )
    print(f"[START] Tryb: {mode_label} | GPU: {torch.cuda.get_device_name(0)}")

    records, images_root = (
        prepare_dry_run_data() if DRY_RUN else prepare_full_train_data()
    )

    records, start_ckpt_idx = apply_resume_filter(records, BASE_OUTPUT_DIR)

    if not records:
        print("[SUKCES] Brak pozostałych obrazów do przetworzenia.")
        return

    model = CLIPModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)
    model.eval()
    CLIPProcessor.from_pretrained(MODEL_ID)

    dataset = VideoGameBunnyDataset(records=records, images_root=images_root)
    collator = ClipCollator(model_id=MODEL_ID)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
        drop_last=False,
    )

    features_buffer: list[list[float]] = []
    metadata_buffer: list[dict[str, Any]] = []
    checkpoint_idx = start_ckpt_idx
    total_processed = 0

    start_time = time.perf_counter()

    for inputs, meta in tqdm(dataloader, desc="Ekstrakcja cech CLIP"):
        if inputs is None or not meta:
            continue

        gpu_inputs = {k: v.to(DEVICE, non_blocking=True) for k, v in inputs.items()}

        with torch.amp.autocast("cuda"), torch.no_grad():
            outputs = model.get_image_features(**gpu_inputs)
            img_feats = (
                outputs if isinstance(outputs, torch.Tensor) else outputs.pooler_output
            )
            img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)

        features_buffer.extend(img_feats.to(torch.float32).cpu().numpy().tolist())
        metadata_buffer.extend(meta)
        total_processed += len(meta)

        if len(features_buffer) >= CHECKPOINT_FREQ:
            save_checkpoint(
                features_buffer, metadata_buffer, checkpoint_idx, BASE_OUTPUT_DIR
            )
            features_buffer.clear()
            metadata_buffer.clear()
            checkpoint_idx += 1

    if features_buffer:
        save_checkpoint(
            features_buffer, metadata_buffer, checkpoint_idx, BASE_OUTPUT_DIR
        )

    elapsed = time.perf_counter() - start_time
    throughput = total_processed / elapsed if elapsed > 0 else 0.0
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)

    print("\n" + "=" * 50)
    print(f"[SUKCES] Wyekstrahowano {total_processed} z {len(records)} kadrów.")
    print(
        f"[METRYKI] Czas obliczeń CLIP: {elapsed:.2f} s | Przepustowość: {throughput:.1f} img/s"
    )
    print(f"[METRYKI] Szczytowe zużycie VRAM: {peak_vram_gb:.2f} GB")
    print("=" * 50)


if __name__ == "__main__":
    main()
