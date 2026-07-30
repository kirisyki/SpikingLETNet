[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$HwinfoExe,
    [string]$HwinfoLog = "",
    [ValidateSet("auto", "ymd", "mdy", "dmy")]
    [string]$DateOrder = "auto",
    [string]$PowerColumn = ""
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path $HwinfoExe -PathType Leaf)) {
    throw "找不到 HWiNFO64.exe: $HwinfoExe"
}
if (-not (Test-Path ".venv\Scripts\python.exe" -PathType Leaf)) {
    throw "请先运行 .\setup_windows.ps1"
}

$signature = Get-AuthenticodeSignature -FilePath $HwinfoExe
if ($signature.Status -ne [System.Management.Automation.SignatureStatus]::Valid) {
    throw "HWiNFO 可执行文件签名无效：$($signature.Status) / $($signature.StatusMessage)"
}
$fileInfo = Get-Item $HwinfoExe
if ($fileInfo.VersionInfo.ProductName -notmatch "HWiNFO") {
    throw "可执行文件产品名不是 HWiNFO：$($fileInfo.VersionInfo.ProductName)"
}
if ($signature.SignerCertificate.Subject -notmatch "REALiX|Martin Malik") {
    throw "签名有效但发布者不是官方 HWiNFO/REALiX：$($signature.SignerCertificate.Subject)"
}


$version = $fileInfo.VersionInfo.FileVersion
$result = [ordered]@{
    checked_at = (Get-Date).ToString("o")
    executable = (Resolve-Path $HwinfoExe).Path
    file_version = $version
    product_name = $fileInfo.VersionInfo.ProductName
    company_name = $fileInfo.VersionInfo.CompanyName
    signature_status = $signature.Status.ToString()
    signer_subject = $signature.SignerCertificate.Subject
    signer_thumbprint = $signature.SignerCertificate.Thumbprint
    secure_boot_or_hvci_was_not_modified = $true
}
New-Item -ItemType Directory -Force "results" | Out-Null
$result | ConvertTo-Json -Depth 4 | Set-Content -Path "results\hwinfo_collector_check.json" -Encoding utf8

Write-Host "HWiNFO 签名检查通过，版本：$version" -ForegroundColor Green

if ($HwinfoLog) {
    if (-not (Test-Path $HwinfoLog -PathType Leaf)) {
        throw "找不到 HWiNFO 日志：$HwinfoLog"
    }
    $arguments = @(
        "scripts\hwinfo_tools.py", "inspect",
        "--log", $HwinfoLog,
        "--date-order", $DateOrder,
        "--output", "results\hwinfo_log_inspection.json"
    )
    if ($PowerColumn) {
        $arguments += @("--power-column", $PowerColumn)
    }
    & ".venv\Scripts\python.exe" @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "HWiNFO 日志结构检查失败。"
    }
}

Write-Host "HWiNFO 采集器检查完成。正式记录时请保持 Sensors-only 日志连续运行。" -ForegroundColor Green
