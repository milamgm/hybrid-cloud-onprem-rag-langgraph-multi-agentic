# AI/API Gateway

Esta carpeta define la primera capa de la arquitectura híbrida. Las aplicaciones
solo conocen una API compatible con OpenAI; la elección del proveedor y los
controles de tráfico permanecen fuera del código de negocio.

| Entorno | Gateway | Objetivo |
| --- | --- | --- |
| Cloud (Azure) | Azure API Management Developer | Laboratorio temporal: autenticación, cuota por token y acceso gestionado a Azure AI Foundry. |
| On-premise | LiteLLM Proxy | Endpoint local único para los servidores de inferencia compatibles con OpenAI. |

## Principios de seguridad

- No guardes secretos en archivos versionados. Copia los archivos `.env.example`
  a `.env` solo en el entorno correspondiente.
- Expón LiteLLM mediante un Ingress o reverse proxy con TLS y autenticación; el
  `docker-compose.yml` de desarrollo lo publica exclusivamente en `127.0.0.1`.
- En Azure, el backend usa Managed Identity. Las aplicaciones se autentican ante
  APIM con una subscription key de alcance mínimo; no reciben la credencial del
  modelo.
- Fija y verifica la imagen de LiteLLM antes de promoverla. El tag elegido está
  versionado; para producción debe sustituirse por un digest aprobado.
- El Bicep crea APIM Developer para que el laboratorio tenga `llm-token-limit`.
  No es un tier de producción y debe eliminarse antes de acabar el crédito Azure.

## Desarrollo local y promoción

LM Studio es el upstream de desarrollo recomendado: inicia su servidor local y
expone la API OpenAI-compatible en `http://127.0.0.1:1234/v1`. Como LiteLLM se
ejecuta dentro de Docker, su configuración debe usar
`http://host.docker.internal:1234/v1` para alcanzar LM Studio en el host.

Para una promoción posterior no cambia la aplicación ni la URL pública de
LiteLLM: se sustituye `ONPREM_UPSTREAM_BASE_URL` por la URL privada de vLLM o
NVIDIA NIM y se actualiza `ONPREM_LITELLM_MODEL`. Esto permite enseñar una ruta
de desarrollo reproducible y una ruta de producción escalable con el mismo
contrato.

## Uso

1. Despliega o integra `azure-apim/main.bicep` en el resource group que contiene
   tu instancia de APIM. Completa los valores de named value que crea el módulo.
   Antes del despliegue, habilita una managed identity en APIM y asígnale el rol
   mínimo necesario sobre Azure AI Foundry (por ejemplo, `Cognitive Services
   OpenAI User`). Crea una subscription distinta por aplicación/entorno.
2. Para on-premise, copia `litellm/.env.example` a `litellm/.env` y arranca el
   perfil con `docker compose --env-file infrastructure/ai-gateway/litellm/.env -f infrastructure/ai-gateway/litellm/docker-compose.yml up -d`.
3. Copia las variables del gateway que corresponda a la raíz `.env` de la
   aplicación. `src.config.config` elige el gateway cuando estas variables están
   presentes.

Los guardrails, DLP y autorización de herramientas se incorporarán como capas
posteriores alrededor de estos mismos puntos de entrada.
