#!/usr/bin/env python3
"""Generic HTTP server for mocking backend APIs.

This module provides a YAML-driven mock HTTP server for testing MCP servers.
It supports route-based request matching and response generation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import yaml

# Types
HeadersType = Dict[str, str]
JSONLike = Any


@dataclass
class Route:
    """Represents a route configuration for the mock server."""

    method: str
    path: str
    status: int
    headers: HeadersType
    body: JSONLike
    body_file: Optional[str] = None


@dataclass
class DefaultResponse:
    """Represents a default response configuration for unmatched routes."""

    status: int
    headers: HeadersType
    body: JSONLike


class RoutesConfig:
    """Configuration for mock server routes loaded from YAML."""

    def __init__(self, routes: List[Route], default: Optional[DefaultResponse]):
        self.routes = routes
        self.default = default

    @staticmethod
    def from_yaml_file(path: str) -> "RoutesConfig":  # pylint: disable=too-many-locals
        """Load route configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        raw_routes: list[dict[str, Any]] = data.get("routes", [])
        routes: list[Route] = []
        config_dir = os.path.dirname(os.path.abspath(path))
        for r in raw_routes:
            method = str(r.get("method", "GET")).upper()
            path_value = str(r.get("path", "/"))
            status = int(r.get("status", 200))
            headers = dict(r.get("headers", {}) or {})
            body = r.get("body", "")
            body_file_value_raw = r.get("body_file")
            body_file_value: Optional[str] = None
            if isinstance(body_file_value_raw, str) and body_file_value_raw.strip():
                candidate = body_file_value_raw.strip()
                if not os.path.isabs(candidate):
                    candidate = os.path.normpath(os.path.join(config_dir, candidate))
                body_file_value = candidate
            routes.append(
                Route(
                    method=method,
                    path=path_value,
                    status=status,
                    headers=headers,
                    body=body,
                    body_file=body_file_value,
                )
            )
        raw_default = data.get("default")
        default: Optional[DefaultResponse] = None
        if raw_default is not None:
            default = DefaultResponse(
                status=int(raw_default.get("status", 404)),
                headers=dict(raw_default.get("headers", {}) or {}),
                body=raw_default.get("body", {"error": "not mocked"}),
            )
        return RoutesConfig(routes=routes, default=default)

    def match(self, method: str, path: str) -> Optional[Route]:
        """Match a request method and path to a configured route."""
        for route in self.routes:
            if route.method == method and route.path == path:
                return route
        return None


class MockRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for mock server that matches routes and returns configured responses."""

    server_version = "InsightsMockHTTP/1.0"

    def _read_body(self) -> bytes:
        """Read request body based on Content-Length header."""
        content_length = self.headers.get("Content-Length")
        if not content_length:
            return b""
        try:
            length = int(content_length)
        except ValueError:
            return b""
        return self.rfile.read(length)

    def _handle(self) -> None:  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        parsed = urlparse(self.path)
        method = self.command.upper()
        path_only = parsed.path
        body_bytes = self._read_body()
        body_preview = body_bytes[:512]

        route = self.server.routes_config.match(method, path_only)  # type: ignore[attr-defined]
        if route is None:
            # Unmatched request
            logging.warning(
                "Unmatched request: method=%s path=%s query=%s body_preview=%s",
                method,
                path_only,
                parsed.query,
                body_preview.decode("utf-8", errors="replace"),
            )
            default: Optional[DefaultResponse] = self.server.routes_config.default  # type: ignore[attr-defined]
            status, headers, body = self._default_response(default)
        else:
            status, headers, body = route.status, route.headers, route.body
            details = ""
            if headers:
                details += f"headers={headers}\n"
            else:
                details += "headers={}\n"
            if body:
                details += f"body={body}\n"
            else:
                details += "body={}\n"
            logging.info("Matched request: method=%s path=%s -> status=%s", method, path_only, status)
            logging.debug("Request %s: details=%s", path_only, details)

        # Serialize body
        payload: bytes
        content_type = headers.get("Content-Type", "")
        file_payload: Optional[bytes] = None
        if route is not None and getattr(route, "body_file", None):
            try:
                with open(route.body_file, "rb") as fh:  # type: ignore[arg-type]
                    file_payload = fh.read()
                if not content_type:
                    if str(route.body_file).endswith(".json"):  # type: ignore[arg-type]
                        content_type = "application/json"
                    else:
                        content_type = "application/octet-stream"
            except OSError as exc:
                logging.error("Failed to read body_file '%s': %s", route.body_file, exc)
                logging.error(
                    "Consider running 'make sync-openapi' to update the mock server with the latest OpenAPI spec."
                )
                file_payload = None
        if file_payload is not None:
            payload = file_payload
        else:
            if isinstance(body, (dict, list)):
                payload = json.dumps(body).encode("utf-8")
                if not content_type:
                    content_type = "application/json"
            elif isinstance(body, (str, bytes)):
                payload = body if isinstance(body, bytes) else body.encode("utf-8")
                if not content_type:
                    content_type = "text/plain; charset=utf-8"
            else:
                payload = str(body).encode("utf-8")
                if not content_type:
                    content_type = "text/plain; charset=utf-8"

        logging.debug("Sending response: content_type=%s\npayload=%s\n", content_type, payload)
        # Send response
        self.send_response(status)
        # Ensure Content-Type is set
        headers_out = dict(headers)
        headers_out["Content-Type"] = content_type
        for k, v in headers_out.items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(payload)

    @staticmethod
    def _default_response(default: Optional[DefaultResponse]) -> Tuple[int, HeadersType, JSONLike]:
        if default is None:
            return 404, {"Content-Type": "application/json"}, {"error": "not mocked"}
        return default.status, default.headers, default.body

    def do_GET(self) -> None:  # pylint: disable=invalid-name  # noqa: N802
        """Handle GET requests."""
        self._handle()

    def do_POST(self) -> None:  # pylint: disable=invalid-name  # noqa: N802
        """Handle POST requests."""
        self._handle()

    def do_PUT(self) -> None:  # pylint: disable=invalid-name  # noqa: N802
        """Handle PUT requests."""
        self._handle()

    def do_DELETE(self) -> None:  # pylint: disable=invalid-name  # noqa: N802
        """Handle DELETE requests."""
        self._handle()

    def log_message(self, fmt: str, *args: Any) -> None:  # pylint: disable=arguments-differ,useless-return
        """Suppress default stderr logging."""
        _ = fmt
        _ = args


def _configure_logging(level: str, log_file: Optional[str]) -> None:
    """Configure logging for the mock server."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = []
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    else:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=handlers,
    )


def _write_port_file(port_file: str, port: int) -> None:
    """Write the server port to a file for test consumers."""
    os.makedirs(os.path.dirname(port_file), exist_ok=True)
    with open(port_file, "w", encoding="utf-8") as f:
        f.write(str(port))
        f.flush()
        os.fsync(f.fileno())


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the mock server."""
    parser = argparse.ArgumentParser(description="Generic YAML-driven mock HTTP server.")
    parser.add_argument("--config", required=True, help="Path to YAML routes config.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=0, help="Bind port (0 for ephemeral).")
    parser.add_argument("--port-file", required=True, help="Write the chosen port to this file.")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    parser.add_argument("--log-file", default=None, help="Write logs to file instead of stdout.")
    return parser.parse_args()


def main() -> None:
    """Main entry point for the mock HTTP server."""
    args = parse_args()
    _configure_logging(args.log_level, args.log_file)
    routes_config = RoutesConfig.from_yaml_file(args.config)

    # Create server; port 0 lets OS choose a free port
    server_address = (args.host, args.port)
    # Prefer ThreadingHTTPServer for concurrent requests during tests
    httpd = ThreadingHTTPServer(server_address, MockRequestHandler)
    # Attach config for handler access
    httpd.routes_config = routes_config  # type: ignore[attr-defined]

    # After binding, get real port and write to file for consumers
    real_port: int
    sock: socket.socket = httpd.socket  # type: ignore[assignment]
    real_port = sock.getsockname()[1]
    try:
        _write_port_file(args.port_file, real_port)
    except OSError as exc:
        logging.error("Failed to write port file '%s': %s", args.port_file, exc)
        sys.exit(1)

    logging.info("Mock server listening on http://%s:%s", args.host, real_port)
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
