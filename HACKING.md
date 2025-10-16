# Insights MCP Contributing Guide

## Run

⚠️ Usually you want to just use the MCP server via a tool like VSCode, Cursor, etc.
so please refer to the [integrations](README.md#integrations) section unless you want to
develop the MCP server.

Also checkout `make help` for the available commands.

## Important notes
* When changing some code you might want to use `make build-prod` so the container is built with
  the upstream container tag and you don't need to change it in your MCP client (like VSCode).

* Make sure you really restart VSCode or Cursor after changing the code, as their "restart" button
  usually doesn't use the newly built container.

* ⚠️ Moreover, when you start VSCode, make sure you hit the `▶️ Start` button of the MCP server,
  **before** you start chatting! Otherwise VSCode _caches_ the tool descriptions and you will
  end up with a chat context with the old tool descriptions!


### Usage

See [usage.md](usage.md) for the usage of the MCP server.

### Using Python directly

#### Option 1: Global CLI tool (recommended for usage)
Install as a global CLI tool (lighter, no development dependencies):

```bash
uv tool install -e .
```

Then run directly:

```bash
insights-mcp sse
```

#### Option 2: Project environment (recommended for development)
Set up the development environment (includes development dependencies for testing, linting, etc.):

```bash
uv sync --all-extras --dev
```

Then run with `uv`:

```bash
uv run insights-mcp sse
```

**Note**: Use Option 2 if you need to run tests, linting, or other development tasks:
```bash
uv run pytest
uv run mypy src/
uv run pylint src/
```

Both approaches will start `insights-mcp` server at http://localhost:9000/sse

For HTTP streaming transport:

```bash
insights-mcp http
```

This will start `insights-mcp` server with HTTP streaming transport at http://localhost:8000/mcp

### Using Podman/Docker

You can also copy the command from the [Makefile]
For SSE mode:
```
make run-sse
```

For HTTP streaming mode:
```
make run-http
```

You can also copy the command from the [Makefile]
For stdio mode:
```
make run-stdio
```

### Additional info

You can set the environment variable `IMAGE_BUILDER_MCP_DISABLE_DESCRIPTION_WATERMARK` to `True` to avoid adding a hint to newly created image builder blueprints.
## Pipelines as Code configuration
To start the PipelineRun, add a new comment in a pull-request with content `/ok-to-test`

If a test fails, add a new comment in a pull-request with content `/retest` to re-run the test.

For more detailed information about running a PipelineRun, please refer to Pipelines as Code documentation [Running the PipelineRun](https://pipelinesascode.com/docs/guide/running/)

To customize the proposed PipelineRuns after merge, please refer to [Build Pipeline customization](https://konflux-ci.dev/docs/how-tos/configuring/)

Please follow the block sequence indentation style introduced by the proposed PipelineRuns YAMLs, or keep using consistent indentation level through your customized PipelineRuns. When different levels are mixed, it will be changed to the proposed style.

## Stage

To use the internal stage server, you need to set the environment variables accordingly:

* `INSIGHTS_BASE_URL`
* `INSIGHTS_STAGE_PROXY_URL`
* `INSIGHTS_STAGE_TOKEN_ENDPOINT`
* and enable with `--stage` flag

The concrete values are considered internal and thus are not documented here.

```json
{
	"servers": {
		"insights-mcp-stage": {
			"type": "stdio",
			"command": "podman",
			"args": [
				"run",
				"--env",
				"INSIGHTS_CLIENT_ID",
				"--env",
				"INSIGHTS_CLIENT_SECRET",
				"--env",
				"INSIGHTS_BASE_URL",
				"--env",
				"INSIGHTS_STAGE_PROXY_URL",
				"--env",
				"INSIGHTS_STAGE_TOKEN_ENDPOINT",
				"--stage",
				"--interactive",
				"--rm",
				"ghcr.io/redhatinsights/insights-mcp:latest",
				"--debug"
			],
			"env": {
				"INSIGHTS_CLIENT_ID": "${input:insights_client_stage_id}",
				"INSIGHTS_CLIENT_SECRET": "${input:insights_client_stage_secret}",
				"INSIGHTS_BASE_URL": "${input:insights_base_url}",
				"INSIGHTS_STAGE_PROXY_URL": "${input:insights_stage_proxy_url}",
				"INSIGHTS_STAGE_TOKEN_ENDPOINT": "${input:insights_stage_token_endpoint}"
			}
		}
  },
  "inputs": [
		{
			"id": "insights_client_stage_id",
			"type": "promptString",
			"description": "Enter the Red Hat Insights Stage Client ID",
			"default": "",
			"password": true
		},
		{
			"id": "insights_client_stage_secret",
			"type": "promptString",
			"description": "Enter the Red Hat Insights Stage Client Secret",
			"default": "",
			"password": true
		},
		{
			"id": "insights_base_url",
			"type": "promptString",
			"description": "Enter the Red Hat Insights Base URL"
		},
		{
			"id": "insights_stage_proxy_url",
			"type": "promptString",
			"description": "Enter the Red Hat Insights Stage Proxy URL"
		},
		{
			"id": "insights_stage_token_endpoint",
			"type": "promptString",
			"description": "Enter the Red Hat Insights Stage Token Endpoint"
		}
	]
}
```
