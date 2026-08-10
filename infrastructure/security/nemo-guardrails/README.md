# NemoGuard JailbreakDetect NIM

The on-premise middleware uses NVIDIA's dedicated `jailbreak detection model`
rail. The NIM is a separate GPU service: it does not use the application LLM
or a self-check prompt.

Copy `.env.example` in this directory to `.env`, set `NGC_API_KEY`, and start:

```bash
docker compose --env-file .env up -d
```

Set the corresponding application variables:

```env
NEMO_JAILBREAK_NIM_BASE_URL=http://localhost:8123/v1/
NVIDIA_API_KEY=<NVIDIA API key>
```

The service is bound to localhost and exposes its readiness probe at
`http://localhost:8123/v1/health/ready`.

## Output content safety

Output moderation uses a second, dedicated NIM: `nvidia/llama-3.1-nemotron-
safety-guard-8b-v3`. Deploy it separately from JailbreakDetect and expose its
OpenAI-compatible endpoint at the value of `NEMO_CONTENT_SAFETY_NIM_BASE_URL`
(the application default is `http://localhost:8124/v1`). It is already defined
as `nemotron-safety-guard` in `docker-compose.yml`. The NeMo output rail
configuration is in `output-validation.yml`; it deliberately does not replace
the local JSON/Pydantic/JSON Schema, citation and Presidio checks.
