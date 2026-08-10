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
