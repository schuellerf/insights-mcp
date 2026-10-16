#!/usr/bin/env python3
"""Generate diagrams from Mermaid source in HACKING.md.

This script extracts Mermaid code blocks from HACKING.md and generates
diagrams (SVG, PNG, etc.) using Podman with the Mermaid CLI container.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MERMAID_CLI_IMAGE = "ghcr.io/mermaid-js/mermaid-cli/mermaid-cli:12.0.0"
# PNG runs use this binary as the container entrypoint.
MERMAID_CLI_BIN = "/home/mermaidcli/node_modules/.bin/mmdc"
# Host temporary directory, mounted as the container HOME for Chromium.
CONTAINER_HOME = "/tmp"
PNG_PLATFORM = "linux/amd64"
PNG_PUPPETEER_CONFIG_NAME = "puppeteer-png.json"
PNG_CSS_NAME = "diagram-png.css"
PNG_PUPPETEER_CONFIG: dict[str, str | list[str]] = {
    "executablePath": "/usr/bin/chromium-browser",
    "args": [
        "--no-sandbox",
        "--disable-gpu",
        "--disable-gpu-compositing",
        "--use-gl=disabled",
    ],
}
# themeCSS for PNG renders. Disables drop-shadow filters on sequence and flowchart shapes.
PNG_DROP_SHADOW_CSS = """\
g rect.rect { filter: none !important; }
rect.actor { filter: none !important; }
.labelBox { filter: none !important; }
.node rect { filter: none !important; }
.node polygon { filter: none !important; }
.node path { filter: none !important; }
.node circle { filter: none !important; }
"""


def extract_mermaid_blocks(markdown_file: Path) -> dict[str, str]:
    """Extract Mermaid code blocks from markdown file.

    Args:
        markdown_file: Path to the markdown file

    Returns:
        Dictionary mapping diagram names to Mermaid source code
    """
    content = markdown_file.read_text(encoding="utf-8")
    mermaid_blocks = {}

    # Pattern to match Mermaid code blocks with optional title
    pattern = r"```mermaid\n(.*?)```"
    matches = re.finditer(pattern, content, re.DOTALL)

    for i, match in enumerate(matches):
        mermaid_code = match.group(1).strip()

        # Extract diagram name from title comment: %% title: diagram-name
        title_pattern = r"%%\s*title:\s*([^\n]+)"
        title_match = re.search(title_pattern, mermaid_code, re.IGNORECASE)
        if not title_match:
            print(
                f"Error: Mermaid diagram {i + 1} is missing '%% title: diagram-name' comment",
                file=sys.stderr,
            )
            sys.exit(1)

        diagram_name = title_match.group(1).strip()
        if not diagram_name:
            print(
                f"Error: Mermaid diagram {i + 1} has empty title in '%% title:' comment",
                file=sys.stderr,
            )
            sys.exit(1)

        mermaid_blocks[diagram_name] = mermaid_code

    return mermaid_blocks


def check_podman() -> bool:
    """Check if podman is available.

    Returns:
        True if podman is available, False otherwise
    """
    return shutil.which("podman") is not None


def amd64_emulation_hint() -> str:
    """Explain a failed linux/amd64 PNG run.

    Returns:
        Hint naming the pinned image, the missing emulation, and HACKING.md.
    """
    return (
        "PNG renders are pinned to linux/amd64 and are not retried on the host architecture. "
        f"PNG generation runs {MERMAID_CLI_IMAGE} as linux/amd64 so the screenshot matches the committed bytes. "
        "Apple Silicon Podman needs amd64 emulation (qemu-user-static inside the podman machine). "
        "See the diagram note under Deployment Flow in HACKING.md. "
        "SVG generation does not need amd64."
    )


def diagram_generation_error(format_type: str, returncode: int, stderr: str) -> str:
    """Explain a failed mermaid-cli run.

    Args:
        format_type: Requested output format, such as ``svg`` or ``png``.
        returncode: Exit status from podman.
        stderr: Captured podman stderr. Included when it reports an exec format error.

    Returns:
        Error text. PNG failures name the amd64 pin and the HACKING.md emulation note.
    """
    message = f"Failed to generate {format_type.upper()}: podman exited with status {returncode}"
    if format_type != "png":
        return message
    if "exec format" in stderr.lower():
        message = f"{message}: {stderr.strip()}"
    return f"{message}. {amd64_emulation_hint()}"


def podman_run_command(docs_dir: Path, output_file: Path, format_type: str, home_dir: Path) -> list[str]:
    """Build the podman argv that renders one diagram.

    SVG runs stay on the host architecture and skip font embedding.
    PNG runs use linux/amd64, ``MERMAID_CLI_BIN``, the software-GL Puppeteer
    config, and the drop-shadow stylesheet.
    ``home_dir`` is mounted as the container HOME.

    Args:
        docs_dir: Host directory mounted at ``/data`` in the container.
        output_file: Host path whose name is the container output file.
        format_type: ``svg`` or ``png``.
        home_dir: Host directory to mount as ``CONTAINER_HOME``.

    Returns:
        Argv for ``podman run``.
    """
    command = [
        "podman",
        "run",
        "--rm",
        "--userns=keep-id",
        f"--user={os.getuid()}",
        "-e",
        f"HOME={CONTAINER_HOME}",
        "-v",
        f"{home_dir.resolve()}:{CONTAINER_HOME}:Z",
    ]
    if format_type == "png":
        command.extend(["--platform", PNG_PLATFORM, "--entrypoint", MERMAID_CLI_BIN])
    command.extend(
        [
            "-v",
            f"{docs_dir.absolute()}:/data:Z",
            MERMAID_CLI_IMAGE,
            "-i",
            "/data/input.mmd",
            "-o",
            f"/data/{output_file.name}",
        ]
    )
    if format_type == "svg":
        command.append("--no-font-embed")
    if format_type == "png":
        command.extend(
            [
                "-p",
                f"/data/{PNG_PUPPETEER_CONFIG_NAME}",
                "-C",
                f"/data/{PNG_CSS_NAME}",
            ]
        )
    return command


def _write_png_render_files(docs_dir: Path) -> list[Path]:
    """Write the puppeteer config and drop-shadow override into the mount.

    Args:
        docs_dir: Directory mounted into the container as ``/data``.

    Returns:
        Paths that the caller must delete after the container exits.
    """
    puppeteer_path = docs_dir / PNG_PUPPETEER_CONFIG_NAME
    css_path = docs_dir / PNG_CSS_NAME
    puppeteer_path.write_text(json.dumps(PNG_PUPPETEER_CONFIG, indent=2) + "\n", encoding="utf-8")
    css_path.write_text(PNG_DROP_SHADOW_CSS, encoding="utf-8")
    return [puppeteer_path, css_path]


def generate_diagram_from_mermaid(mermaid_code: str, output_file: Path, docs_dir: Path, format_type: str) -> None:
    """Generate diagram from Mermaid code using Podman with Mermaid CLI container.

    Args:
        mermaid_code: Mermaid diagram source code
        output_file: Path where the output file should be written
        docs_dir: Directory containing docs (used for podman mount)
        format_type: Output format (e.g., 'svg', 'png')

    Raises:
        RuntimeError: If podman exits unsuccessfully. PNG failures include the amd64 emulation hint.
        FileNotFoundError: If podman is not available
    """
    if not check_podman():
        raise FileNotFoundError("podman is not available. Please install podman to generate diagrams.")

    # Write input file directly to docs directory
    input_file = docs_dir / "input.mmd"
    input_file.write_text(mermaid_code, encoding="utf-8")

    docs_dir.mkdir(parents=True, exist_ok=True)
    extra_files: list[Path] = []
    if format_type == "png":
        extra_files.extend(_write_png_render_files(docs_dir))

    try:
        with tempfile.TemporaryDirectory(prefix="insights-mcp-diagram-") as diagram_home:
            cmd = podman_run_command(docs_dir, output_file, format_type, Path(diagram_home))
            subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Post-process SVG to make box border match background color
        # This makes the border effectively invisible
        # Only apply post-processing to SVG files
        if format_type == "svg" and output_file.exists():
            _make_box_border_match_background(output_file, rgb_color=(225, 245, 255))
    except subprocess.CalledProcessError as e:
        if e.stdout:
            print(e.stdout, file=sys.stderr)
        if e.stderr:
            print(e.stderr, file=sys.stderr)
        stderr_text = e.stderr or ""
        raise RuntimeError(diagram_generation_error(format_type, e.returncode, stderr_text)) from e
    finally:
        for path in [input_file, *extra_files]:
            if path.exists():
                path.unlink()


def _make_box_border_match_background(svg_file: Path, rgb_color: tuple[int, int, int]) -> None:
    """Post-process SVG to make box border color match background color.

    This is a workaround to make the box border color match the background color.
    The SVG generated by Mermaid CLI has a border color that is not the same as the background color.
    This is a workaround to make the border color match the background color
    as there is no way to set the border color in the Mermaid code or remove the border completely.

    Args:
        svg_file: Path to the SVG file to modify
        rgb_color: RGB tuple (r, g, b) representing the background color
    """
    try:
        # Read SVG as text for easier manipulation
        content = svg_file.read_text(encoding="utf-8")

        # Convert RGB to hex color
        hex_color = f"#{rgb_color[0]:02x}{rgb_color[1]:02x}{rgb_color[2]:02x}"
        rgb_str = f"rgb({rgb_color[0]}, {rgb_color[1]}, {rgb_color[2]})"

        # Find rect elements with our background color and update their stroke
        # Pattern: rect with fill="rgb(225, 245, 255)" and stroke="..."
        pattern = rf'(<[^>]*rect[^>]*fill=["\']?{re.escape(rgb_str)}["\']?[^>]*stroke=["\']?)([^"\'>]+)(["\']?[^>]*>)'

        def replace_stroke(match):
            return f"{match.group(1)}{hex_color}{match.group(3)}"

        content = re.sub(pattern, replace_stroke, content)

        # Also handle stroke in style attributes
        # Pattern: style="...stroke:rgb(...)..."
        style_pattern = r'(style=["\'])([^"\']*stroke:\s*)[^;)]+([;"\'])'

        def replace_style_stroke(match):
            return f"{match.group(1)}{match.group(2)}{hex_color}{match.group(3)}"

        content = re.sub(style_pattern, replace_style_stroke, content)

        # Ensure file ends with newline (pre-commit requirement)
        if not content.endswith("\n"):
            content += "\n"

        # Write modified SVG
        svg_file.write_text(content, encoding="utf-8")
    except Exception as e:
        # If post-processing fails, just log and continue with original SVG
        print(f"Warning: Could not post-process SVG border color: {e}", file=sys.stderr)


def main():
    """Main function to generate diagrams."""
    parser = argparse.ArgumentParser(
        description="Generate diagrams from Mermaid source in HACKING.md",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--format",
        type=str,
        default="svg,png",
        help="Comma-separated list of output formats (default: svg,png)",
    )
    args = parser.parse_args()

    # Parse formats
    formats = [f.strip().lower() for f in args.format.split(",") if f.strip()]
    if not formats:
        print("Error: At least one format must be specified", file=sys.stderr)
        sys.exit(1)

    # Validate formats
    valid_formats = {"svg", "png"}
    invalid_formats = set(formats) - valid_formats
    if invalid_formats:
        print(
            f"Error: Invalid format(s): {', '.join(invalid_formats)}. Valid formats: {', '.join(valid_formats)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Get project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    hacking_md = project_root / "HACKING.md"
    docs_dir = project_root / "docs"

    if not hacking_md.exists():
        print(f"Error: {hacking_md} not found", file=sys.stderr)
        sys.exit(1)

    # Create docs directory if it doesn't exist
    docs_dir.mkdir(exist_ok=True)

    # Extract Mermaid blocks
    mermaid_blocks = extract_mermaid_blocks(hacking_md)

    if not mermaid_blocks:
        print("No Mermaid diagrams found in HACKING.md", file=sys.stderr)
        sys.exit(1)

    # Check if podman is available
    if not check_podman():
        print(
            "Error: podman is not available. Please install podman to generate diagrams.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Generate diagrams in all requested formats
    for diagram_name, mermaid_code in mermaid_blocks.items():
        for format_type in formats:
            print(f"Generating {diagram_name}.{format_type}...")
            try:
                output_file = docs_dir / f"{diagram_name}.{format_type}"
                generate_diagram_from_mermaid(mermaid_code, output_file, docs_dir, format_type)
                print(f"✓ Saved to {output_file}")
            except (FileNotFoundError, RuntimeError) as e:
                print(f"✗ Error: {e}", file=sys.stderr)
                sys.exit(1)


if __name__ == "__main__":
    main()
