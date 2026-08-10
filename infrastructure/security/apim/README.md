# APIM / Entra ID tool operations

APIM enforces access to remote or sensitive tools at the HTTP boundary. Attach
`tool-operation-policy.xml` to each tool operation and define the named values:

- `tool-entraid-openid-config-url`
- `tool-api-audience`
- `tool-required-role`

Create one APIM operation per remote tool and assign the smallest Entra ID app
role required, for example `tool.web.search` or `tool.sensitive.execute`.
Human approval is not delegated to APIM: the application must pass a recorded
approval ID to the policy decision point before it sends a mutating request.
