#!/usr/bin/env python3
"""Sync OpenAPI specs from URLs into the repository.

This script is used to run "offline" tests with the mock server
(after the specs are synced into the repository).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, List

import yaml


@dataclass
class OpenAPISpec:
    name: str
    url: str
    dst: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync OpenAPI specs from URLs into the repository.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML mapping file (e.g., openapi-sources.yaml).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on any download error (default: continue on errors).",
    )
    return parser.parse_args()


def load_specs(config_path: str) -> List[OpenAPISpec]:
    with open(config_path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}
    items = data.get("openapi", [])
    specs: list[OpenAPISpec] = []
    base_dir = os.path.dirname(os.path.abspath(config_path))
    for item in items:
        name = str(item.get("name", "")).strip()
        url = str(item.get("url", "")).strip()
        dst_raw = str(item.get("dst", "")).strip()
        if not name or not url or not dst_raw:
            print(f"Skipping invalid spec entry: {item}", file=sys.stderr)
            continue
        dst = dst_raw if os.path.isabs(dst_raw) else os.path.normpath(os.path.join(base_dir, "..", dst_raw))
        specs.append(OpenAPISpec(name=name, url=url, dst=dst))
    return specs


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def download_to_file(url: str, dst: str) -> None:
    ensure_parent_dir(dst)
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "insights-mcp-openapi-sync/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:  # nosec - intended network access
            with open(tmp_path, "wb") as out_f:
                shutil.copyfileobj(resp, out_f)
        shutil.move(tmp_path, dst)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass


def sync_spec(spec: OpenAPISpec, strict: bool) -> bool:
    try:
        print(f"[{spec.name}] Fetching {spec.url} -> {spec.dst}")
        download_to_file(spec.url, spec.dst)
        return True
    except (urllib.error.URLError, OSError) as exc:
        print(f"[{spec.name}] WARN: download failed: {exc}", file=sys.stderr)
        return not strict


def main() -> None:
    args = parse_args()
    specs = load_specs(args.config)
    if not specs:
        print("No specs found in configuration.", file=sys.stderr)
        sys.exit(0)
    success_all = True
    for spec in specs:
        ok = sync_spec(spec, args.strict)
        success_all = success_all and ok
    sys.exit(0 if success_all else 1)


if __name__ == "__main__":
    main()
