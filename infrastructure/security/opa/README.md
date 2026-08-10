# Open Policy Agent

OPA is the on-premise policy decision point for agent tool calls. The policy
receives the tool name, arguments, caller identity, roles, data classification,
requested limit and approval ID. It returns `allow`, `requires_approval` and a
reason without executing the tool.

Start it on the production host:

```bash
docker compose up -d
```

The decision endpoint is `POST /v1/data/onyx/tools/decision` and is only bound
to localhost. Configure the application with:

```env
OPA_URL=http://localhost:8181
```
