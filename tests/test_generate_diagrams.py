"""Tests for architecture diagram podman command construction."""

from pathlib import Path

from generate_diagrams import (
    CONTAINER_HOME,
    MERMAID_CLI_BIN,
    PNG_CSS_NAME,
    PNG_DROP_SHADOW_CSS,
    PNG_PUPPETEER_CONFIG_NAME,
    diagram_generation_error,
    podman_run_command,
)


def test_png_stylesheet_disables_mermaid_drop_shadows() -> None:
    """PNG CSS turns off the box, actor, and flowchart filters Mermaid 12 paints."""
    assert "g rect.rect { filter: none !important; }" in PNG_DROP_SHADOW_CSS
    assert "rect.actor { filter: none !important; }" in PNG_DROP_SHADOW_CSS
    assert ".node rect { filter: none !important; }" in PNG_DROP_SHADOW_CSS


def test_svg_podman_command_pins_amd64_without_png_flags(tmp_path: Path) -> None:
    """SVG runs use linux/amd64 and do not pass PNG Chromium flags."""
    docs_dir = tmp_path / "docs"
    home_dir = tmp_path / "diagram-home"
    output = docs_dir / "architecture-structure.svg"
    command = podman_run_command(docs_dir, output, "svg", home_dir)

    assert command[command.index("--platform") + 1] == "linux/amd64"
    assert "--entrypoint" not in command
    assert command[command.index("-e") + 1] == f"HOME={CONTAINER_HOME}"
    assert f"{home_dir.resolve()}:{CONTAINER_HOME}:Z" in command
    assert "--no-font-embed" in command
    assert "-p" not in command
    assert "-C" not in command
    assert command[command.index("-o") + 1] == "/data/architecture-structure.svg"


def test_png_podman_command_pins_amd64_software_render(tmp_path: Path) -> None:
    """PNG runs use linux/amd64 and the software-GL puppeteer config."""
    docs_dir = tmp_path / "docs"
    home_dir = tmp_path / "diagram-home"
    output = docs_dir / "architecture-deployment.png"
    command = podman_run_command(docs_dir, output, "png", home_dir)

    assert command[command.index("-e") + 1] == f"HOME={CONTAINER_HOME}"
    assert f"{home_dir.resolve()}:{CONTAINER_HOME}:Z" in command
    assert command[command.index("--platform") + 1] == "linux/amd64"
    assert command[command.index("--entrypoint") + 1] == MERMAID_CLI_BIN
    assert command.count("-p") == 1
    assert command[command.index("-p") + 1] == f"/data/{PNG_PUPPETEER_CONFIG_NAME}"
    assert command[command.index("-C") + 1] == f"/data/{PNG_CSS_NAME}"
    assert "--no-font-embed" not in command


def test_png_failure_names_missing_amd64_emulation() -> None:
    """An exec format error on PNG points at HACKING.md and does not offer an arm64 retry."""
    message = diagram_generation_error("png", 1, "exec format error")

    assert "exec format error" in message
    assert "linux/amd64" in message
    assert "not retried on the host architecture" in message
    assert "qemu-user-static" in message
    assert "HACKING.md" in message


def test_svg_exec_format_error_names_missing_amd64_emulation() -> None:
    """An exec format error on SVG points at HACKING.md and does not offer an arm64 retry."""
    message = diagram_generation_error("svg", 2, "exec format error")

    assert "exec format error" in message
    assert "linux/amd64" in message
    assert "not retried on the host architecture" in message
    assert "qemu-user-static" in message
    assert "HACKING.md" in message


def test_svg_failure_without_exec_format_omits_amd64_hint() -> None:
    """Other SVG failures stay a plain podman exit and do not mention amd64 emulation."""
    message = diagram_generation_error("svg", 2, "mermaid parse error")

    assert message == "Failed to generate SVG: podman exited with status 2"
    assert "HACKING.md" not in message
