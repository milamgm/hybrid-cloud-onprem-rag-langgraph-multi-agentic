@description('Globally unique name for the temporary API Management service.')
param apimServiceName string = 'onyxagentic-apim-lab'

@description('Email displayed as the publisher contact in API Management.')
param publisherEmail string

@description('OpenAI-compatible Azure OpenAI endpoint, including /openai/v1 and no trailing slash.')
param azureAiBackendUrl string = 'https://onyxagentic.openai.azure.com/openai/v1'

@description('Name of the existing Microsoft Foundry account that hosts the model deployment.')
param foundryAccountName string = 'onyxagentic'

@description('Token quota per minute for each APIM subscription.')
param tokensPerMinute int = 5000

@description('Display name for the API exposed to application workloads.')
param apiDisplayName string = 'Agentic AI Gateway (lab)'

resource foundryAccount 'Microsoft.CognitiveServices/accounts@2024-10-01' existing = {
  name: foundryAccountName
}

// Developer is the lowest fixed-capacity tier that supports llm-token-limit.
// It is explicitly an evaluation tier; delete this resource before the trial ends.
resource apim 'Microsoft.ApiManagement/service@2024-05-01' = {
  name: apimServiceName
  location: resourceGroup().location
  sku: {
    name: 'Developer'
    capacity: 1
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    publisherEmail: publisherEmail
    publisherName: 'Onyx Agentic AI Lab'
  }
}

resource apimFoundryRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: foundryAccount
  name: guid(foundryAccount.id, apim.id, 'CognitiveServicesOpenAIUser')
  properties: {
    principalId: apim.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'
    )
  }
}

resource azureAiBackendUrlValue 'Microsoft.ApiManagement/service/namedValues@2024-05-01' = {
  parent: apim
  name: 'agentic-ai-backend-url'
  properties: {
    displayName: 'agentic-ai-backend-url'
    secret: false
    value: azureAiBackendUrl
  }
}

resource tokensPerMinuteValue 'Microsoft.ApiManagement/service/namedValues@2024-05-01' = {
  parent: apim
  name: 'agentic-ai-tokens-per-minute'
  properties: {
    displayName: 'agentic-ai-tokens-per-minute'
    secret: false
    value: string(tokensPerMinute)
  }
}

resource aiGatewayApi 'Microsoft.ApiManagement/service/apis@2024-05-01' = {
  parent: apim
  name: 'agentic-ai'
  properties: {
    displayName: apiDisplayName
    path: 'ai/v1'
    protocols: [
      'https'
    ]
    subscriptionRequired: true
  }
}

resource chatCompletionsOperation 'Microsoft.ApiManagement/service/apis/operations@2024-05-01' = {
  parent: aiGatewayApi
  name: 'chat-completions'
  properties: {
    displayName: 'Create chat completion'
    method: 'POST'
    urlTemplate: '/chat/completions'
    templateParameters: []
  }
}

resource chatCompletionsPolicy 'Microsoft.ApiManagement/service/apis/operations/policies@2024-05-01' = {
  parent: chatCompletionsOperation
  name: 'policy'
  properties: {
    format: 'rawxml'
    value: '''<policies>
  <inbound>
    <base />
    <llm-token-limit counter-key="@(context.Subscription.Id)" tokens-per-minute="{{agentic-ai-tokens-per-minute}}" estimate-prompt-tokens="true" />
    <set-backend-service base-url="{{agentic-ai-backend-url}}" />
    <authentication-managed-identity resource="https://cognitiveservices.azure.com" />
  </inbound>
  <backend><base /></backend>
  <outbound><base /></outbound>
  <on-error><base /></on-error>
</policies>'''
  }
  dependsOn: [
    apimFoundryRole
  ]
}

resource embeddingsOperation 'Microsoft.ApiManagement/service/apis/operations@2024-05-01' = {
  parent: aiGatewayApi
  name: 'embeddings'
  properties: {
    displayName: 'Create embeddings'
    method: 'POST'
    urlTemplate: '/embeddings'
    templateParameters: []
  }
}

resource embeddingsPolicy 'Microsoft.ApiManagement/service/apis/operations/policies@2024-05-01' = {
  parent: embeddingsOperation
  name: 'policy'
  properties: {
    format: 'rawxml'
    value: '''<policies>
  <inbound>
    <base />
    <llm-token-limit counter-key="@(context.Subscription.Id)" tokens-per-minute="{{agentic-ai-tokens-per-minute}}" estimate-prompt-tokens="true" />
    <set-backend-service base-url="{{agentic-ai-backend-url}}" />
    <authentication-managed-identity resource="https://cognitiveservices.azure.com" />
  </inbound>
  <backend><base /></backend>
  <outbound><base /></outbound>
  <on-error><base /></on-error>
</policies>'''
  }
  dependsOn: [
    apimFoundryRole
  ]
}

output gatewayBaseUrl string = 'https://${apimServiceName}.azure-api.net/ai/v1'
output apiId string = aiGatewayApi.name
output apimResourceId string = apim.id
