[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory)]
    [ValidateNotNullOrEmpty()]
    [string]$ResourceGroup,

    [Parameter(Mandatory)]
    [ValidateNotNullOrEmpty()]
    [string]$ApimServiceName,

    [Parameter(Mandatory)]
    [ValidatePattern('^[^@\s]+@[^@\s]+\.[^@\s]+$')]
    [string]$PublisherEmail,

    [Parameter(Mandatory)]
    [ValidatePattern('^https://.+/openai/v1$')]
    [string]$AzureAiBackendUrl,

    [ValidateRange(1, 10000000)]
    [int]$TokensPerMinute = 60000
)

$ErrorActionPreference = 'Stop'

$templateFile = Join-Path $PSScriptRoot 'main.bicep'
$deploymentName = "agentic-ai-gateway-$(Get-Date -Format 'yyyyMMddHHmmss')"

if ($PSCmdlet.ShouldProcess($ResourceGroup, "Deploy $deploymentName")) {
    az deployment group create `
        --resource-group $ResourceGroup `
        --name $deploymentName `
        --template-file $templateFile `
        --parameters "apimServiceName=$ApimServiceName" "publisherEmail=$PublisherEmail" "azureAiBackendUrl=$AzureAiBackendUrl" "tokensPerMinute=$TokensPerMinute"
}
