param(
    [Parameter(Mandatory = $true)] [string] $DocxPath,
    [Parameter(Mandatory = $true)] [string] $PdfPath,
    [Parameter(Mandatory = $true)] [string] $LogPath
)

$ErrorActionPreference = 'Stop'
$word = $null
$doc = $null
try {
    'starting' | Set-Content -LiteralPath $LogPath -Encoding UTF8
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    $word.Options.SaveNormalPrompt = $false
    $doc = $word.Documents.Open($DocxPath, $false, $false)
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Fields.Update() | Out-Null
    $doc.Repaginate()
    $doc.Save()
    ('updated pages=' + $doc.ComputeStatistics(2)) | Set-Content -LiteralPath $LogPath -Encoding UTF8
    $doc.ExportAsFixedFormat($PdfPath, 17, $false, 0, 0, 1, 9999, 0, $true, $true, 1, $true, $true, $false)
    'exported' | Set-Content -LiteralPath $LogPath -Encoding UTF8
}
catch {
    ('error: ' + $_.Exception.ToString()) | Set-Content -LiteralPath $LogPath -Encoding UTF8
    exit 1
}
finally {
    if ($doc -ne $null) { $doc.Close($false) }
    if ($word -ne $null) { $word.Quit() }
}

