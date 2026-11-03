"""Train a seed-family-isolated GRU for per-step drift-onset detection."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


SEED = 42
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[^\w\s]", re.UNICODE)
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
DECISION_LABELS = {"KILL_SESSION": 1, "WARN": 1, "CONTINUE": 0}


@dataclass
class Episode:
    record_id: str
    family_id: str
    source: str
    category: str
    decision: str
    onset_step: int | None
    texts: list[str]
    step_numbers: list[int]
    labels: list[int]


def parse_json_value(value: Any, field: str, record_id: str) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{record_id}: invalid JSON in {field}: {exc}") from exc
    return value


def render_action_step(goal: str, step: dict[str, Any]) -> str:
    action = step.get("action") or {}
    if not isinstance(action, dict):
        action = {"raw": str(action)}
    arguments = action.get("args")
    observation = step.get("observation")
    if isinstance(observation, (dict, list)):
        observation = json.dumps(observation, ensure_ascii=False, sort_keys=True)
    return (
        f"Original goal: {goal}. "
        f"Action type: {action.get('type') or ''}. "
        f"Tool: {action.get('name') or ''}. "
        f"Tool class: {action.get('class') or ''}. "
        f"Arguments: {json.dumps(arguments, ensure_ascii=False, sort_keys=True)}. "
        f"Action: {action.get('raw') or ''}. "
        f"Observation: {observation or ''}."
    )


def input_fingerprint(record: dict[str, Any]) -> str:
    trajectory = parse_json_value(record.get("trajectory"), "trajectory", str(record.get("id")))
    if not isinstance(trajectory, list):
        raise ValueError(f"{record.get('id')}: trajectory must be a list")
    steps = []
    for raw_step in trajectory:
        step = parse_json_value(raw_step, "trajectory step", str(record.get("id")))
        if not isinstance(step, dict):
            raise ValueError(f"{record.get('id')}: trajectory step must be an object")
        steps.append(
            render_action_step(str(record.get("declared_goal") or ""), step)
        )
    payload = json.dumps(steps, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_records(path: Path, source: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    records = pd.read_parquet(path).to_dict(orient="records")
    for row in records:
        row["_source"] = source
    return records


def ensure_benchmark_files(data_dir: Path, download: bool) -> tuple[Path, Path]:
    raw_dir = data_dir / "raw"
    gold_path = raw_dir / "agentic_redteam_gold" / "data" / "gold.parquet"
    augmented_path = raw_dir / "agentic_redteam_augmented" / "data" / "augmented.parquet"
    missing = [path for path in (gold_path, augmented_path) if not path.is_file()]
    if missing and not download:
        raise FileNotFoundError(
            "Required benchmark files are missing. Run without --no-download or download "
            f"the Hugging Face splits. Missing: {', '.join(map(str, missing))}"
        )
    if gold_path not in missing and augmented_path not in missing:
        return gold_path, augmented_path

    gold_path.parent.mkdir(parents=True, exist_ok=True)
    augmented_path.parent.mkdir(parents=True, exist_ok=True)
    for target, filename in (
        (gold_path, "data/gold.parquet"),
        (augmented_path, "data/augmented.parquet"),
    ):
        if target.is_file():
            continue
        downloaded = hf_hub_download(
            repo_id="jash-ai/agentic-redteam-benchmark",
            filename=filename,
            repo_type="dataset",
            local_dir=str(raw_dir / ("agentic_redteam_" + target.parent.parent.name.removeprefix("agentic_redteam_"))),
        )
        downloaded_path = Path(downloaded)
        if downloaded_path.resolve() != target.resolve():
            target.write_bytes(downloaded_path.read_bytes())
    return gold_path, augmented_path


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, first: str, second: str) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root != second_root:
            low, high = sorted((first_root, second_root))
            self.parent[high] = low


def make_family_assignments(
    gold_rows: list[dict[str, Any]],
    augmented_rows: list[dict[str, Any]],
) -> tuple[dict[str, str], dict[str, Any]]:
    gold_by_id = {str(row["id"]): row for row in gold_rows}
    if len(gold_by_id) != len(gold_rows):
        raise ValueError("Gold dataset contains duplicate record IDs")

    family_for_record: dict[str, str] = {}
    union = UnionFind()
    fingerprint_owner: dict[str, str] = {}
    fingerprint_counts = Counter()

    for row in gold_rows:
        record_id = str(row["id"])
        family = f"seed:{record_id}"
        union.add(family)
        family_for_record[f"gold:{record_id}"] = family
        fingerprint = input_fingerprint(row)
        fingerprint_counts[fingerprint] += 1
        if fingerprint in fingerprint_owner:
            union.union(family, fingerprint_owner[fingerprint])
        else:
            fingerprint_owner[fingerprint] = family

    for row in augmented_rows:
        record_id = str(row["id"])
        seed_id = row.get("seed_id")
        if not seed_id:
            provenance = parse_json_value(
                row.get("provenance"), "provenance", record_id
            )
            seed_id = provenance.get("seed_id") if isinstance(provenance, dict) else None
        if not seed_id or str(seed_id) not in gold_by_id:
            raise ValueError(
                f"Augmented record {record_id} has no valid gold seed_id: {seed_id!r}"
            )
        family = f"seed:{seed_id}"
        union.add(family)
        family_for_record[f"augmented:{record_id}"] = family
        fingerprint = input_fingerprint(row)
        fingerprint_counts[fingerprint] += 1
        if fingerprint in fingerprint_owner:
            union.union(family, fingerprint_owner[fingerprint])
        else:
            fingerprint_owner[fingerprint] = family

    family_for_record = {
        record_key: union.find(family)
        for record_key, family in family_for_record.items()
    }

    gold_families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in gold_rows:
        family = family_for_record[f"gold:{row['id']}"]
        gold_families[family].append(row)

    for family, members in gold_families.items():
        labels = {DECISION_LABELS.get(str(row.get("expected_decision"))) for row in members}
        categories = {str(row.get("category") or "unknown") for row in members}
        if len(labels) != 1 or None in labels:
            raise ValueError(f"Exact duplicate family {family} has conflicting gold decisions")
        if len(categories) != 1:
            raise ValueError(f"Exact duplicate family {family} spans categories")

    fingerprint_cluster_count = sum(count > 1 for count in fingerprint_counts.values())
    augmented_family_count = len(
        {family_for_record[f"augmented:{row['id']}"] for row in augmented_rows}
    )
    summary = {
        "gold_records": len(gold_rows),
        "augmented_records": len(augmented_rows),
        "gold_seed_families_after_exact_dedup_grouping": len(gold_families),
        "augmented_seed_families": augmented_family_count,
        "exact_duplicate_trajectory_clusters": fingerprint_cluster_count,
        "gold_seed_ids_missing_augmented_variants": len(
            set(gold_by_id)
            - {
                str(row.get("seed_id"))
                for row in augmented_rows
                if row.get("seed_id") is not None
            }
        ),
    }
    return family_for_record, summary


def split_seed_families(
    gold_rows: list[dict[str, Any]],
    family_for_record: dict[str, str],
    seed: int,
) -> dict[str, set[str]]:
    strata: dict[tuple[str, int], set[str]] = defaultdict(set)
    seen: set[tuple[str, str]] = set()
    for row in gold_rows:
        family = family_for_record[f"gold:{row['id']}"]
        key = (family, str(row["id"]))
        if key in seen:
            continue
        seen.add(key)
        decision = str(row.get("expected_decision"))
        if decision not in DECISION_LABELS:
            raise ValueError(f"{row['id']}: unknown expected_decision {decision!r}")
        strata[(str(row.get("category") or "unknown"), DECISION_LABELS[decision])].add(family)

    rng = random.Random(seed)
    assigned: dict[str, set[str]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    for families in strata.values():
        ordered = sorted(families)
        rng.shuffle(ordered)
        count = len(ordered)
        if count >= 3:
            test_count = max(1, round(count * 0.15))
            validation_count = max(1, round(count * 0.15))
            while test_count + validation_count >= count:
                if test_count >= validation_count and test_count > 1:
                    test_count -= 1
                elif validation_count > 1:
                    validation_count -= 1
                else:
                    break
        else:
            test_count = 0
            validation_count = 0
        assigned["test"].update(ordered[:test_count])
        assigned["validation"].update(
            ordered[test_count : test_count + validation_count]
        )
        assigned["train"].update(ordered[test_count + validation_count :])

    if any(not assigned[name] for name in assigned):
        raise ValueError("Seed-family splitting produced an empty split")
    if (
        assigned["train"] & assigned["validation"]
        or assigned["train"] & assigned["test"]
        or assigned["validation"] & assigned["test"]
    ):
        raise AssertionError("Seed-family split groups overlap")
    return assigned


def build_episodes(
    rows: list[dict[str, Any]],
    source: str,
    family_for_record: dict[str, str],
) -> list[Episode]:
    episodes: list[Episode] = []
    for row in rows:
        record_id = str(row["id"])
        decision = str(row.get("expected_decision"))
        if decision not in DECISION_LABELS:
            raise ValueError(f"{record_id}: unknown expected_decision {decision!r}")
        trajectory = parse_json_value(row.get("trajectory"), "trajectory", record_id)
        if not isinstance(trajectory, list) or not trajectory:
            raise ValueError(f"{record_id}: trajectory is empty or not a list")

        parsed_steps: list[dict[str, Any]] = []
        for raw_step in trajectory:
            step = parse_json_value(raw_step, "trajectory step", record_id)
            if not isinstance(step, dict):
                raise ValueError(f"{record_id}: trajectory step must be an object")
            parsed_steps.append(step)
        parsed_steps.sort(key=lambda step: int(step.get("index", 0)))

        step_numbers = [int(step.get("index", index)) for index, step in enumerate(parsed_steps)]
        if len(step_numbers) != len(set(step_numbers)):
            raise ValueError(f"{record_id}: duplicate step indexes")
        goal = str(row.get("declared_goal") or "")
        texts = [render_action_step(goal, step) for step in parsed_steps]

        if DECISION_LABELS[decision] == 0:
            onset = None
            targets = [0] * len(parsed_steps)
        else:
            raw_onset = row.get("drift_step")
            if raw_onset is None:
                ground_truth = parse_json_value(
                    row.get("ground_truth"), "ground_truth", record_id
                )
                raw_onset = (
                    ground_truth.get("drift_step")
                    if isinstance(ground_truth, dict)
                    else None
                )
            if raw_onset is None:
                raise ValueError(f"{record_id}: positive drift decision has no drift_step")
            onset = int(raw_onset)
            if onset not in step_numbers:
                raise ValueError(
                    f"{record_id}: drift_step {onset} is absent from trajectory indexes"
                )
            targets = [int(step_number >= onset) for step_number in step_numbers]

        family = family_for_record[f"{source}:{record_id}"]
        episodes.append(
            Episode(
                record_id=record_id,
                family_id=family,
                source=source,
                category=str(row.get("category") or "unknown"),
                decision=decision,
                onset_step=onset,
                texts=texts,
                step_numbers=step_numbers,
                labels=targets,
            )
        )
    return episodes


def deduplicate_episodes(episodes: list[Episode]) -> tuple[list[Episode], int]:
    unique: list[Episode] = []
    seen: dict[str, Episode] = {}
    for episode in episodes:
        fingerprint = hashlib.sha256(
            json.dumps(episode.texts, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        previous = seen.get(fingerprint)
        if previous is None:
            seen[fingerprint] = episode
            unique.append(episode)
            continue
        if (
            previous.step_numbers != episode.step_numbers
            or previous.labels != episode.labels
        ):
            raise ValueError(
                "Exact duplicate trajectories have conflicting step indexes or labels: "
                f"{previous.record_id} and {episode.record_id}"
            )
    return unique, len(episodes) - len(unique)


def make_vocab(episodes: list[Episode], max_vocab: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for episode in episodes:
        for text in episode.texts:
            counts.update(TOKEN_PATTERN.findall(text.lower()))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, _ in counts.most_common(max_vocab - len(vocab)):
        vocab[token] = len(vocab)
    return vocab


def encode_episode(
    episode: Episode, vocab: dict[str, int], max_tokens_per_step: int
) -> list[list[int]]:
    unknown = vocab[UNK_TOKEN]
    return [
        [
            vocab.get(token, unknown)
            for token in TOKEN_PATTERN.findall(text.lower())[:max_tokens_per_step]
        ]
        or [unknown]
        for text in episode.texts
    ]


class EpisodeDataset(Dataset):
    def __init__(
        self,
        episodes: list[Episode],
        vocab: dict[str, int],
        family_sizes: Counter[str],
        max_tokens_per_step: int,
    ) -> None:
        self.episodes = episodes
        self.encoded = [
            encode_episode(episode, vocab, max_tokens_per_step)
            for episode in episodes
        ]
        self.family_weights = [
            1.0 / family_sizes[episode.family_id] for episode in episodes
        ]

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode = self.episodes[index]
        return {
            "tokens": self.encoded[index],
            "labels": episode.labels,
            "step_numbers": episode.step_numbers,
            "family_weight": self.family_weights[index],
        }


def collate_episodes(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    batch_size = len(batch)
    max_steps = max(len(item["tokens"]) for item in batch)
    max_tokens = max(len(step) for item in batch for step in item["tokens"])
    tokens = torch.zeros((batch_size, max_steps, max_tokens), dtype=torch.long)
    labels = torch.zeros((batch_size, max_steps), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_steps), dtype=torch.bool)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    family_weights = torch.zeros(batch_size, dtype=torch.float32)
    for batch_index, item in enumerate(batch):
        lengths[batch_index] = len(item["tokens"])
        family_weights[batch_index] = item["family_weight"]
        for step_index, token_ids in enumerate(item["tokens"]):
            tokens[batch_index, step_index, : len(token_ids)] = torch.tensor(
                token_ids, dtype=torch.long
            )
            labels[batch_index, step_index] = item["labels"][step_index]
            mask[batch_index, step_index] = True
    return {
        "tokens": tokens,
        "labels": labels,
        "mask": mask,
        "lengths": lengths,
        "family_weights": family_weights,
    }


class GoalDriftGRU(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 48,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.step_projection = nn.Linear(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, token_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        token_mask = token_ids.ne(0).unsqueeze(-1)
        pooled = (embedded * token_mask).sum(dim=2) / token_mask.sum(dim=2).clamp(min=1)
        step_vectors = self.dropout(torch.relu(self.step_projection(pooled)))
        packed = pack_padded_sequence(
            step_vectors,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.gru(packed)
        padded, _ = pad_packed_sequence(
            encoded, batch_first=True, total_length=token_ids.shape[1]
        )
        return self.classifier(self.dropout(padded)).squeeze(-1)


def load_trained_model(
    checkpoint_path: Path,
) -> tuple[GoalDriftGRU, dict[str, int], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid GRU checkpoint: {checkpoint_path}")
    model_config = checkpoint.get("model_config")
    vocab = checkpoint.get("vocab")
    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_config, dict) or not isinstance(vocab, dict):
        raise ValueError(f"GRU checkpoint is missing model configuration or vocabulary: {checkpoint_path}")
    if not isinstance(model_state, dict):
        raise ValueError(f"GRU checkpoint is missing model weights: {checkpoint_path}")
    model = GoalDriftGRU(
        vocab_size=int(model_config["vocab_size"]),
        embedding_dim=int(model_config["embedding_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        dropout=float(model_config["dropout"]),
    )
    model.load_state_dict(model_state)
    model.eval()
    return model, vocab, model_config


def predict_trajectory_steps(
    model: GoalDriftGRU,
    vocab: dict[str, int],
    original_goal: str,
    trajectory: list[dict[str, Any]],
    max_tokens_per_step: int = 256,
    threshold: float | None = None,
) -> list[dict[str, float | int]]:
    if not trajectory:
        raise ValueError("Cannot score an empty trajectory")
    ordered_steps = sorted(
        trajectory,
        key=lambda step: int(step.get("index", 0)),
    )
    step_numbers = [
        int(step.get("index", index)) for index, step in enumerate(ordered_steps)
    ]
    if len(step_numbers) != len(set(step_numbers)):
        raise ValueError("Trajectory contains duplicate step indexes")
    texts = [render_action_step(original_goal, step) for step in ordered_steps]
    episode = Episode(
        record_id="inference",
        family_id="inference",
        source="inference",
        category="unknown",
        decision="unknown",
        onset_step=None,
        texts=texts,
        step_numbers=step_numbers,
        labels=[0] * len(texts),
    )
    dataset = EpisodeDataset(
        [episode],
        vocab,
        Counter({"inference": 1}),
        max_tokens_per_step,
    )
    probabilities = predict_probabilities(model, dataset, batch_size=1)[0]
    return [
        {
            "step": int(step),
            "drift_probability": float(probability),
            "drift_detected": bool(
                threshold is not None and probability >= threshold
            ),
        }
        for step, probability in zip(step_numbers, probabilities)
    ]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def predict_probabilities(
    model: GoalDriftGRU,
    dataset: EpisodeDataset,
    batch_size: int,
) -> list[np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_episodes,
    )
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["tokens"], batch["lengths"])
            probabilities = torch.sigmoid(logits).cpu().numpy()
            mask = batch["mask"].cpu().numpy()
            for row, row_mask in zip(probabilities, mask):
                predictions.append(row[row_mask])
    return predictions


def batch_loss(
    model: GoalDriftGRU,
    batch: dict[str, torch.Tensor],
    positive_weight: float,
) -> torch.Tensor:
    logits = model(batch["tokens"], batch["lengths"])
    criterion = nn.BCEWithLogitsLoss(
        reduction="none",
        pos_weight=torch.tensor(positive_weight, dtype=torch.float32),
    )
    losses = criterion(logits, batch["labels"])
    mask = batch["mask"].float()
    per_episode = (losses * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    weights = batch["family_weights"]
    return (per_episode * weights).sum() / weights.sum().clamp(min=1e-8)


def evaluate_loss(
    model: GoalDriftGRU,
    dataset: EpisodeDataset,
    batch_size: int,
    positive_weight: float,
) -> float:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_episodes,
    )
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            losses.append(float(batch_loss(model, batch, positive_weight).item()))
    return float(np.mean(losses))


def episode_detection(
    episode: Episode, probabilities: np.ndarray, threshold: float, tolerance: int
) -> dict[str, Any]:
    detected = [
        step
        for step, probability in zip(episode.step_numbers, probabilities)
        if probability >= threshold
    ]
    predicted_onset = min(detected) if detected else None
    is_positive = episode.onset_step is not None
    exact = is_positive and predicted_onset == episode.onset_step
    within = (
        is_positive
        and predicted_onset is not None
        and episode.onset_step <= predicted_onset <= episode.onset_step + tolerance
    )
    early = (
        is_positive
        and predicted_onset is not None
        and predicted_onset < episode.onset_step
    )
    return {
        "record_id": episode.record_id,
        "family_id": episode.family_id,
        "category": episode.category,
        "decision": episode.decision,
        "true_onset_step": episode.onset_step,
        "predicted_onset_step": predicted_onset,
        "exact_onset_hit": bool(exact),
        "within_tolerance_hit": bool(within),
        "early_alarm": bool(early),
        "delay_steps": (
            int(predicted_onset - episode.onset_step) if within else None
        ),
        "per_step_probabilities": [
            {"step": int(step), "probability": float(probability)}
            for step, probability in zip(episode.step_numbers, probabilities)
        ],
    }


def event_metrics(
    episodes: list[Episode],
    predictions: list[np.ndarray],
    threshold: float,
    tolerance: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details = [
        episode_detection(episode, probabilities, threshold, tolerance)
        for episode, probabilities in zip(episodes, predictions)
    ]
    tp = sum(row["within_tolerance_hit"] for row in details)
    fp = sum(
        row["predicted_onset_step"] is not None and not row["within_tolerance_hit"]
        for row in details
    )
    fn = sum(row["true_onset_step"] is not None and not row["within_tolerance_hit"] for row in details)
    tn = sum(
        row["true_onset_step"] is None and row["predicted_onset_step"] is None
        for row in details
    )
    clean = [row for row in details if row["true_onset_step"] is None]
    positive = [row for row in details if row["true_onset_step"] is not None]
    delays = [row["delay_steps"] for row in details if row["delay_steps"] is not None]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return (
        {
            "trajectories": len(episodes),
            "drift_trajectories": len(positive),
            "fully_aligned_trajectories": len(clean),
            "threshold": float(threshold),
            "onset_tolerance_steps": tolerance,
            "caught_exact_onset": int(sum(row["exact_onset_hit"] for row in details)),
            "caught_within_tolerance": int(tp),
            "early_false_alarm_on_drift_trajectory": int(
                sum(row["early_alarm"] for row in details)
            ),
            "missed_or_late_drift_trajectories": int(fn),
            "false_alarm_on_aligned_trajectories": int(
                sum(
                    row["true_onset_step"] is None
                    and row["predicted_onset_step"] is not None
                    for row in details
                )
            ),
            "aligned_trajectory_false_alarm_rate": (
                float(
                    sum(
                        row["true_onset_step"] is None
                        and row["predicted_onset_step"] is not None
                        for row in details
                    )
                    / len(clean)
                )
                if clean
                else None
            ),
            "event_precision": float(precision),
            "event_recall_with_tolerance": float(recall),
            "event_f1_with_tolerance": (
                float(2 * precision * recall / (precision + recall))
                if precision + recall
                else 0.0
            ),
            "median_detection_delay_steps": (
                float(np.median(delays)) if delays else None
            ),
            "event_confusion_counts": {
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
            },
        },
        details,
    )


def choose_event_threshold(
    episodes: list[Episode],
    predictions: list[np.ndarray],
    tolerance: int,
) -> tuple[float, dict[str, Any]]:
    all_scores = np.concatenate(predictions)
    candidates = np.unique(
        np.concatenate(
            (
                np.asarray([0.0, 0.5, 1.0]),
                all_scores,
                np.asarray([np.nextafter(float(all_scores.max()), np.inf)]),
            )
        )
    )
    evaluated = [
        (event_metrics(episodes, predictions, float(threshold), tolerance)[0], float(threshold))
        for threshold in candidates
    ]
    best_f1 = max(result["event_f1_with_tolerance"] for result, _ in evaluated)
    best = [
        (result, threshold)
        for result, threshold in evaluated
        if result["event_f1_with_tolerance"] == best_f1
    ]
    result, threshold = max(best, key=lambda item: item[1])
    return threshold, result


def step_metrics(
    episodes: list[Episode],
    predictions: list[np.ndarray],
    threshold: float,
) -> dict[str, Any]:
    y_true = np.asarray([label for episode in episodes for label in episode.labels])
    scores = np.concatenate(predictions)
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    metrics: dict[str, Any] = {
        "steps": int(len(y_true)),
        "positive_steps": int(y_true.sum()),
        "negative_steps": int(len(y_true) - y_true.sum()),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "confusion_counts": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        metrics["average_precision"] = float(average_precision_score(y_true, scores))
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
    return metrics


def train_epoch(
    model: GoalDriftGRU,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    positive_weight: float,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = batch_loss(model, batch, positive_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses))


def train_time_only_baseline(
    train_episodes: list[Episode],
    validation_episodes: list[Episode],
    test_episodes: list[Episode],
    tolerance: int,
) -> dict[str, Any]:
    def features(episodes: list[Episode]) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(
            [[step] for episode in episodes for step in episode.step_numbers],
            dtype=np.float64,
        )
        y = np.asarray([label for episode in episodes for label in episode.labels])
        return x, y

    x_train, y_train = features(train_episodes)
    x_validation, _ = features(validation_episodes)
    x_test, _ = features(test_episodes)
    model = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=SEED, solver="liblinear"
    )
    model.fit(x_train, y_train)

    def episode_probabilities(
        episodes: list[Episode], x_values: np.ndarray
    ) -> list[np.ndarray]:
        raw = model.predict_proba(x_values)[:, 1]
        result: list[np.ndarray] = []
        offset = 0
        for episode in episodes:
            result.append(raw[offset : offset + len(episode.step_numbers)])
            offset += len(episode.step_numbers)
        return result

    val_probs = episode_probabilities(validation_episodes, x_validation)
    threshold, _ = choose_event_threshold(
        validation_episodes, val_probs, tolerance
    )
    test_probs = episode_probabilities(test_episodes, x_test)
    event, _ = event_metrics(test_episodes, test_probs, threshold, tolerance)
    return {
        "validation_selected_threshold": threshold,
        "test_event_metrics": event,
        "test_step_metrics": step_metrics(test_episodes, test_probs, threshold),
    }


def describe_split(episodes: list[Episode]) -> dict[str, Any]:
    return {
        "episodes": len(episodes),
        "seed_families": len({episode.family_id for episode in episodes}),
        "source_counts": dict(Counter(episode.source for episode in episodes)),
        "decisions": dict(Counter(episode.decision for episode in episodes)),
        "categories": dict(Counter(episode.category for episode in episodes)),
        "steps": sum(len(episode.step_numbers) for episode in episodes),
        "positive_steps": sum(sum(episode.labels) for episode in episodes),
        "negative_steps": sum(
            len(episode.labels) - sum(episode.labels) for episode in episodes
        ),
        "step_lengths": dict(Counter(len(episode.step_numbers) for episode in episodes)),
    }


def run(
    data_dir: Path,
    output_dir: Path,
    seed: int = SEED,
    max_epochs: int = 50,
    patience: int = 8,
    batch_size: int = 32,
    download: bool = True,
) -> dict[str, Any]:
    seed_everything(seed)
    gold_path, augmented_path = ensure_benchmark_files(data_dir, download)
    gold_rows = load_records(gold_path, "gold")
    augmented_rows = load_records(augmented_path, "augmented")

    if any(row.get("tier") != "gold" for row in gold_rows):
        raise ValueError("Expected only tier=gold in the gold parquet")
    if any(row.get("tier") != "augmented" for row in augmented_rows):
        raise ValueError("Expected only tier=augmented in the augmented parquet")
    provenance_reviewed = []
    for row in augmented_rows:
        provenance = parse_json_value(row.get("provenance"), "provenance", str(row["id"]))
        provenance_reviewed.append(
            bool(provenance.get("human_reviewed"))
            if isinstance(provenance, dict)
            else False
        )

    family_for_record, family_audit = make_family_assignments(
        gold_rows, augmented_rows
    )
    split_families = split_seed_families(gold_rows, family_for_record, seed)
    augmented_episodes = build_episodes(
        augmented_rows, "augmented", family_for_record
    )
    gold_episodes = build_episodes(gold_rows, "gold", family_for_record)
    augmented_episodes, augmented_duplicates_removed = deduplicate_episodes(
        augmented_episodes
    )
    gold_episodes, gold_duplicates_removed = deduplicate_episodes(gold_episodes)
    family_audit["exact_duplicate_rows_removed"] = {
        "augmented": augmented_duplicates_removed,
        "gold": gold_duplicates_removed,
    }

    train_episodes = [
        episode
        for episode in augmented_episodes
        if episode.family_id in split_families["train"]
    ]
    validation_episodes = [
        episode
        for episode in augmented_episodes
        if episode.family_id in split_families["validation"]
    ]
    test_episodes = [
        episode for episode in gold_episodes if episode.family_id in split_families["test"]
    ]
    if not train_episodes or not validation_episodes or not test_episodes:
        raise ValueError("A data split has no episodes after seed-family filtering")
    if not any(any(episode.labels) for episode in train_episodes):
        raise ValueError("Training split has no positive drift steps")
    if not any(not any(episode.labels) for episode in train_episodes):
        raise ValueError("Training split has no fully aligned trajectories")

    split_family_sets = {
        "train": {episode.family_id for episode in train_episodes},
        "validation": {episode.family_id for episode in validation_episodes},
        "test": {episode.family_id for episode in test_episodes},
    }
    if (
        split_family_sets["train"] & split_family_sets["validation"]
        or split_family_sets["train"] & split_family_sets["test"]
        or split_family_sets["validation"] & split_family_sets["test"]
    ):
        raise AssertionError("Seed families cross data splits")

    vocab = make_vocab(train_episodes, max_vocab=30_000)
    max_tokens_per_step = 256
    train_family_sizes = Counter(episode.family_id for episode in train_episodes)
    train_dataset = EpisodeDataset(
        train_episodes, vocab, train_family_sizes, max_tokens_per_step
    )
    validation_dataset = EpisodeDataset(
        validation_episodes,
        vocab,
        Counter(episode.family_id for episode in validation_episodes),
        max_tokens_per_step,
    )
    test_dataset = EpisodeDataset(
        test_episodes,
        vocab,
        Counter(episode.family_id for episode in test_episodes),
        max_tokens_per_step,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_episodes,
        generator=generator,
    )

    positive_steps = sum(sum(episode.labels) for episode in train_episodes)
    negative_steps = sum(
        len(episode.labels) - sum(episode.labels) for episode in train_episodes
    )
    positive_weight = float(negative_steps / max(positive_steps, 1))
    model = GoalDriftGRU(len(vocab))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    training_history: list[dict[str, float | int]] = []
    for epoch in range(1, max_epochs + 1):
        training_loss = train_epoch(
            model, train_loader, optimizer, positive_weight
        )
        validation_loss = evaluate_loss(
            model,
            validation_dataset,
            batch_size,
            positive_weight,
        )
        training_history.append(
            {
                "epoch": epoch,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint")
    model.load_state_dict(best_state)

    validation_probabilities = predict_probabilities(
        model, validation_dataset, batch_size
    )
    tolerance = 1
    threshold, validation_event_metrics = choose_event_threshold(
        validation_episodes, validation_probabilities, tolerance
    )
    test_probabilities = predict_probabilities(model, test_dataset, batch_size)
    test_event, test_details = event_metrics(
        test_episodes, test_probabilities, threshold, tolerance
    )
    test_step = step_metrics(test_episodes, test_probabilities, threshold)

    time_only = train_time_only_baseline(
        train_episodes, validation_episodes, test_episodes, tolerance
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "goalguard_gru.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "vocab": vocab,
            "model_config": {
                "vocab_size": len(vocab),
                "embedding_dim": 48,
                "hidden_dim": 64,
                "dropout": 0.2,
                "max_tokens_per_step": max_tokens_per_step,
                "threshold": threshold,
                "onset_tolerance_steps": tolerance,
                "seed": seed,
            },
        },
        model_path,
    )
    details_path = output_dir / "heldout_gold_predictions.jsonl"
    with details_path.open("w", encoding="utf-8") as file:
        for detail in test_details:
            file.write(json.dumps(detail, ensure_ascii=False) + "\n")

    report = {
        "experiment": "GoalGuard GRU per-step drift-onset detection",
        "generated_from": {
            "gold_file": str(gold_path),
            "augmented_file": str(augmented_path),
            "hugging_face_dataset": "jash-ai/agentic-redteam-benchmark",
        },
        "label_mapping": {
            "KILL_SESSION": "drift; steps >= drift_step are positive",
            "WARN": "drift; steps >= drift_step are positive",
            "CONTINUE": "fully aligned negative-control trajectory; all steps are negative",
            "important": (
                "CONTINUE records can still have a populated drift_step field. The decision label "
                "takes precedence: do not treat that field as a true onset for CONTINUE."
            ),
        },
        "provenance": {
            "augmented_trajectories": len(augmented_rows),
            "augmented_human_reviewed": int(sum(provenance_reviewed)),
            "augmented_not_human_reviewed": int(
                len(provenance_reviewed) - sum(provenance_reviewed)
            ),
            "gold_test_records_are_independently_reviewed_by_goalguard": False,
            "source_card_reports_gold_single_author_and_IAA_pending": True,
            "caveat": (
                "Augmented variants are synthetic and share ancestry with gold seeds. They are "
                "training data only; all descendants of held-out test seed families are excluded "
                "from both train and validation."
            ),
        },
        "split_method": {
            "grouping_unit": "seed family, with exact duplicate trajectories joined",
            "training": "augmented rows from train seed families",
            "validation": "augmented rows from validation seed families",
            "test": "gold rows from held-out test seed families",
            "fractions_target": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "seed": seed,
            "family_audit": family_audit,
            "family_counts": {
                name: len(values) for name, values in split_families.items()
            },
            "family_ids": {
                name: sorted(values) for name, values in split_families.items()
            },
        },
        "split_distributions": {
            "train": describe_split(train_episodes),
            "validation": describe_split(validation_episodes),
            "test": describe_split(test_episodes),
        },
        "model": {
            "architecture": (
                "word-token embedding mean-pooling within each observed step, "
                "one-layer unidirectional GRU over steps, per-step binary output"
            ),
            "embedding_dim": 48,
            "hidden_dim": 64,
            "dropout": 0.2,
            "vocabulary_size": len(vocab),
            "input_features": [
                "declared goal",
                "tool action type and name",
                "tool class and arguments",
                "visible tool-call text",
                "tool observation through current step",
            ],
            "excluded_features": [
                "private thought text",
                "ground_truth rationale",
                "expected_decision as a feature",
                "drift_step as a feature",
                "metadata and provenance",
                "future steps",
            ],
            "loss": (
                "per-trajectory average binary cross-entropy, positive-class weighted, "
                "with inverse variant-count family weighting"
            ),
            "early_stopping": {
                "criterion": "minimum validation loss",
                "best_epoch": best_epoch,
                "epochs_run": len(training_history),
                "best_validation_loss": best_validation_loss,
            },
        },
        "threshold_selection": {
            "split": "validation augmented seed families",
            "objective": (
                "maximize trajectory-level event F1 for the first alarm in "
                "[drift_step, drift_step + 1]; highest threshold breaks ties"
            ),
            "tolerance_steps": tolerance,
            "threshold": threshold,
            "validation_event_metrics": validation_event_metrics,
        },
        "test": {
            "source": "gold split; seed families unseen in train and validation",
            "event_metrics": test_event,
            "step_metrics": test_step,
            "predictions_file": str(details_path),
        },
        "time_only_baseline": time_only,
        "comparison": {
            "gru_minus_time_only_event_f1": float(
                test_event["event_f1_with_tolerance"]
                - time_only["test_event_metrics"]["event_f1_with_tolerance"]
            ),
            "note": (
                "If time-only performs as well as the GRU, the onset label distribution may "
                "support a timing shortcut rather than content-based detection."
            ),
        },
        "training_history": training_history,
        "artifacts": {
            "model": str(model_path),
            "test_predictions": str(details_path),
        },
        "limitations": [
            "Only 513 gold seed families exist; augmented variants are not independent trajectories.",
            "The augmented split is machine-generated and the source reports human_reviewed=false.",
            "The gold test source is single-author and not independently GoalGuard-reviewed.",
            "This predicts the current/onset step from the prefix through that observed step; it does not forecast future drift before evidence appears.",
            "Benchmark and local-agent distributions differ; a held-out benchmark score does not prove real-agent detection.",
            "Near-duplicate semantic scenarios across different seed IDs may still cross splits.",
        ],
    }
    metrics_path = output_dir / "gru_metrics.json"
    metrics_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("goalguard_data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("goalguard_data") / "gru_experiment"
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    report = run(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        download=not args.no_download,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
