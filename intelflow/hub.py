"""
Model Hub and Download Manager for iNPU.
Downloads and caches INT4 OpenVINO models from Hugging Face Hub.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from huggingface_hub import snapshot_download
from rich.console import Console

from intelflow.config import MODEL_DIR, MODEL_REGISTRY, resolve_model_tag

console = Console()


def validate_repo_id(repo_id: str) -> str:
    """Validate and sanitize a repository ID to prevent path traversal."""
    if not repo_id or not isinstance(repo_id, str):
        raise ValueError("Invalid model repository ID: must be a non-empty string")

    clean_id = repo_id.strip()
    if ".." in clean_id or "\0" in clean_id or clean_id.startswith("/") or clean_id.startswith("\\"):
        raise ValueError(f"Path traversal detected in model identifier: '{clean_id}'")

    if not re.match(r"^[a-zA-Z0-9_.:-]+(/[a-zA-Z0-9_.:-]+)?$", clean_id):
        raise ValueError(f"Invalid model repository format: '{clean_id}'. Expected 'name' or 'org/model'.")

    return clean_id


class ModelHub:
    @staticmethod
    def get_local_path_for_repo(repo_id: str) -> Path:
        """Map repo_id (e.g. 'OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov') to local folder."""
        clean_id = validate_repo_id(repo_id)
        safe_name = clean_id.replace("/", "--")
        primary = (MODEL_DIR / safe_name).resolve()
        model_dir_res = MODEL_DIR.resolve()
        legacy_dir = (Path.home() / ".cache" / "intelflowlm" / "models").resolve()

        if primary.exists():
            if not primary.is_relative_to(model_dir_res):
                raise ValueError(f"Path traversal violation: target directory outside model cache sandbox")
            return primary

        legacy = legacy_dir / safe_name
        if legacy.exists():
            if not legacy.is_relative_to(legacy_dir):
                raise ValueError(f"Path traversal violation: legacy target directory outside sandbox")
            return legacy

        if not primary.is_relative_to(model_dir_res):
            raise ValueError(f"Path traversal violation: target directory outside model cache sandbox")
        return primary

    @classmethod
    def is_downloaded(cls, tag_or_repo: str) -> bool:
        """Check if model files already exist locally."""
        repo_id = resolve_model_tag(tag_or_repo)
        local_path = cls.get_local_path_for_repo(repo_id)
        if not local_path.is_dir():
            return False
        # Check for essential OpenVINO files
        has_xml = any(local_path.glob("*.xml"))
        has_bin = any(local_path.glob("*.bin"))
        return has_xml and has_bin

    @classmethod
    def pull(cls, tag_or_repo: str, force: bool = False) -> Path:
        """Download model from Hugging Face."""
        repo_id = resolve_model_tag(tag_or_repo)
        target_dir = cls.get_local_path_for_repo(repo_id)

        if cls.is_downloaded(tag_or_repo) and not force:
            console.print(f"[green]✔ Model already cached locally at:[/green] {target_dir}")
            return target_dir

        console.print(f"[bold cyan]📥 Pulling model:[/bold cyan] {repo_id}")
        console.print(f"[dim]Destination: {target_dir}[/dim]")

        # Download OpenVINO IR weights and tokenizer configs
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            allow_patterns=["*.xml", "*.bin", "*.json", "*.txt", "*.model", "tokenizer*"],
            resume_download=True,
        )

        console.print(f"[bold green]✔ Successfully downloaded {repo_id}![/bold green]")
        return target_dir

    @classmethod
    def list_installed(cls) -> List[Dict[str, Any]]:
        """List all locally downloaded models with disk usage."""
        installed = []
        candidate_dirs = [MODEL_DIR]
        legacy_dir = Path.home() / ".cache" / "intelflowlm" / "models"
        if legacy_dir.exists() and legacy_dir != MODEL_DIR:
            candidate_dirs.append(legacy_dir)

        seen_repos = set()
        for base_dir in candidate_dirs:
            if not base_dir.exists():
                continue
            for path in base_dir.iterdir():
                if path.is_dir() and any(path.glob("*.xml")):
                    repo_id = path.name.replace("--", "/")
                    if repo_id in seen_repos:
                        continue
                    seen_repos.add(repo_id)
                    total_bytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                    total_mb = total_bytes / (1024 * 1024)

                    # Reverse lookup tag if known
                    matched_tags = [t for t, m in MODEL_REGISTRY.items() if m.get("repo_id") == repo_id]
                    primary_tag = matched_tags[0] if matched_tags else repo_id

                    installed.append({
                        "tag": primary_tag,
                        "repo_id": repo_id,
                        "path": str(path),
                        "size_mb": round(total_mb, 1)
                    })
        return installed

    @classmethod
    def remove(cls, tag_or_repo: str) -> bool:
        """Delete model from local storage."""
        import shutil
        repo_id = resolve_model_tag(tag_or_repo)
        target_dir = cls.get_local_path_for_repo(repo_id).resolve()
        model_dir_res = MODEL_DIR.resolve()
        legacy_dir = (Path.home() / ".cache" / "intelflowlm" / "models").resolve()

        # Enforce target_dir is strictly a child of model cache directory, not the directory itself
        is_in_primary = target_dir.is_relative_to(model_dir_res) and target_dir != model_dir_res
        is_in_legacy = target_dir.is_relative_to(legacy_dir) and target_dir != legacy_dir

        if not (is_in_primary or is_in_legacy):
            raise ValueError(f"Security error: cannot remove path outside model cache sandbox: {target_dir}")

        if target_dir.exists() and target_dir.is_dir():
            shutil.rmtree(target_dir)
            return True
        return False

    @classmethod
    def check(cls, tag_or_repo: str) -> Dict[str, Any]:
        """Verify the integrity of a downloaded OpenVINO model."""
        repo_id = resolve_model_tag(tag_or_repo)
        target_dir = cls.get_local_path_for_repo(repo_id)
        if not target_dir.exists() or not target_dir.is_dir():
            return {
                "tag": tag_or_repo,
                "repo_id": repo_id,
                "installed": False,
                "valid": False,
                "error": f"Model directory not found at {target_dir}",
                "files": [],
                "size_mb": 0.0,
            }

        files = [f.name for f in target_dir.glob("*") if f.is_file()]
        xml_files = list(target_dir.glob("*.xml"))
        bin_files = list(target_dir.glob("*.bin"))

        has_model_ir = any("openvino" in f.name.lower() or "model" in f.name.lower() for f in xml_files)
        has_weights = len(bin_files) > 0
        has_tokenizer = any("tokenizer" in f.lower() for f in files) or any("vocab" in f.lower() for f in files)

        total_bytes = sum(f.stat().st_size for f in target_dir.rglob("*") if f.is_file())
        size_mb = round(total_bytes / (1024 * 1024), 2)

        valid = bool(xml_files and has_weights)
        return {
            "tag": tag_or_repo,
            "repo_id": repo_id,
            "installed": True,
            "valid": valid,
            "has_model_ir": has_model_ir,
            "has_weights": has_weights,
            "has_tokenizer": has_tokenizer,
            "size_mb": size_mb,
            "file_count": len(files),
            "files": sorted(files),
        }

    @classmethod
    def list_catalog(cls) -> Dict[str, Dict[str, Any]]:
        """Return the curated registry of verified models."""
        return MODEL_REGISTRY
