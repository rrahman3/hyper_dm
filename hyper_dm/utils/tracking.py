"""Lightweight MLflow experiment-tracking wrapper.

Usage
-----
In training code::

    from hyper_dm.utils.tracking import RunTracker

    tracker = RunTracker(cfg)           # reads tracking section from config
    tracker.log_params(cfg)             # flat-log entire config dict
    for ep in range(epochs):
        ...
        tracker.log_metrics({"loss": 0.1, "psnr": 30.0}, step=ep)
        tracker.log_checkpoint(path)    # log model artefact
    tracker.end()

Every run gets a unique ``run_id`` (timestamp-based) used to create
isolated checkpoint / output directories.  When MLflow is enabled the
run_id matches MLflow's run ID; otherwise a local timestamp is used.

When ``tracking.enabled`` is ``false`` (the default), every call is a no-op so
existing code keeps working without MLflow installed.
"""

from __future__ import annotations

import datetime
import pathlib
import tempfile
from typing import Any


def _flatten_dict(d: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict into ``{'a.b.c': value}`` pairs for MLflow params."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        else:
            out[key] = v
    return out


def _make_run_id() -> str:
    """Create a short, filesystem-safe run identifier."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class RunTracker:
    """Thin wrapper around MLflow that degrades gracefully if disabled.

    Attributes
    ----------
    run_id : str
        Unique identifier for this run (MLflow run ID or local timestamp).
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        tcfg = cfg.get("tracking", {})
        self.enabled: bool = tcfg.get("enabled", False)
        self._run = None

        # Always create a run_id, even without MLflow — used for directory naming
        self.run_id: str = _make_run_id()

        if not self.enabled:
            return

        try:
            import mlflow  # noqa: F811
        except ImportError:
            print("[tracking] mlflow not installed – tracking disabled.")
            self.enabled = False
            return

        experiment = tcfg.get("experiment_name", "hyper-dm")
        tracking_uri = tcfg.get("tracking_uri", "mlruns")  # local folder default

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

        run_name = tcfg.get("run_name", None)
        tags = tcfg.get("tags", {})
        self._run = mlflow.start_run(run_name=run_name, tags=tags)
        self._mlflow = mlflow
        # Use MLflow's own run ID so artifacts link correctly
        self.run_id = self._run.info.run_id
        print(f"[tracking] MLflow run started: {self.run_id}")

    # ── directory helpers ────────────────────────────────────────

    def make_run_dir(self, base: str | pathlib.Path) -> pathlib.Path:
        """Create ``<base>/<run_id>/`` and return it.

        Guarantees each experiment run writes to its own isolated folder.
        """
        d = pathlib.Path(base) / self.run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── logging helpers ──────────────────────────────────────────

    def log_params(self, cfg: dict[str, Any]) -> None:
        """Log a (possibly nested) config dict as MLflow parameters."""
        if not self.enabled:
            return
        flat = _flatten_dict(cfg)
        # MLflow has a 500-param hard limit; truncate long values
        for k, v in flat.items():
            self._mlflow.log_param(k, str(v)[:250])

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log a dict of scalar metrics at a given step."""
        if not self.enabled:
            return
        self._mlflow.log_metrics(metrics, step=step)

    def log_checkpoint(self, state_dict: dict, filename: str) -> None:
        """Save a model checkpoint directly to MLflow (no persistent local copy).

        When MLflow is disabled, this is a no-op — use :meth:`save_checkpoint`
        to get a local fallback path.
        """
        if not self.enabled:
            return
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / filename
            import torch
            torch.save(state_dict, path)
            self._mlflow.log_artifact(str(path), artifact_path="checkpoints")

    def save_checkpoint(
        self, state_dict: dict, filename: str, fallback_dir: pathlib.Path
    ) -> None:
        """Industry-standard checkpoint save.

        * MLflow ON  → upload to artifact store only (no disk copy).
        * MLflow OFF → save to ``fallback_dir/<filename>`` on disk.
        """
        if self.enabled:
            self.log_checkpoint(state_dict, filename)
        else:
            fallback_dir.mkdir(parents=True, exist_ok=True)
            import torch
            torch.save(state_dict, fallback_dir / filename)

    def log_artifact(self, path: str | pathlib.Path, artifact_path: str | None = None) -> None:
        """Log a single file as an MLflow artefact."""
        if not self.enabled:
            return
        self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_artifact_dir(self, local_dir: str | pathlib.Path, artifact_path: str | None = None) -> None:
        """Log an entire directory of files as MLflow artefacts."""
        if not self.enabled:
            return
        self._mlflow.log_artifacts(str(local_dir), artifact_path=artifact_path)

    def log_image(
        self,
        array,
        filename: str,
        artifact_path: str = "samples",
        cmap: str = "gray",
    ) -> None:
        """Save a 2-D numpy array as a PNG and log to MLflow.

        Used to store a small number of visual samples for quick QA in the UI.
        """
        if not self.enabled:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as tmp:
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(array, cmap=cmap)
            ax.axis("off")
            path = pathlib.Path(tmp) / filename
            fig.savefig(path, bbox_inches="tight", pad_inches=0.05, dpi=100)
            plt.close(fig)
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_text(self, text: str, filename: str = "notes.txt") -> None:
        """Log arbitrary text (e.g. a summary) as an artefact."""
        if not self.enabled:
            return
        tmp = pathlib.Path(filename)
        tmp.write_text(text)
        self._mlflow.log_artifact(str(tmp))
        tmp.unlink(missing_ok=True)

    def log_config_yaml(self, cfg: dict[str, Any]) -> None:
        """Save the full config YAML as an artefact for exact reproducibility."""
        if not self.enabled:
            return
        import yaml
        tmp = pathlib.Path("config_snapshot.yaml")
        tmp.write_text(yaml.dump(cfg, default_flow_style=False))
        self._mlflow.log_artifact(str(tmp), artifact_path="config")
        tmp.unlink(missing_ok=True)

    def set_tag(self, key: str, value: str) -> None:
        if not self.enabled:
            return
        self._mlflow.set_tag(key, value)

    def end(self) -> None:
        """End the current MLflow run."""
        if not self.enabled:
            return
        self._mlflow.end_run()
        print("[tracking] MLflow run ended.")
