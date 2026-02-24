param(
    [string]$AudiverisPath = "C:\Program Files\Audiveris\Audiveris.exe",
    [string]$PdfDir = "C:\Users\Admin\dev\bach_gen\data\imslp_pdfs",
    [string]$OutDir = "C:\Users\Admin\dev\bach_gen\data\imslp_musicxml",
    [string]$CondaEnv = "bach-gen",
    [switch]$PlainXml,
    [switch]$SkipOmr,
    [switch]$SkipExtract
)

$ErrorActionPreference = "Stop"

function Get-MusicXmlFiles {
    param([string]$Dir)
    if (-not (Test-Path $Dir)) {
        return @()
    }
    return Get-ChildItem -Path $Dir -Recurse -File |
        Where-Object { $_.Extension -in @(".xml", ".musicxml", ".mxl") }
}

function Resolve-AudiverisJava {
    param([string]$AudiverisExe)
    $installDir = Split-Path -Parent $AudiverisExe
    $javaExe = Join-Path $installDir "runtime\bin\java.exe"
    $appDir = Join-Path $installDir "app"
    if ((Test-Path $javaExe) -and (Test-Path $appDir)) {
        return @{
            JavaExe = $javaExe
            AppDir = $appDir
        }
    }
    return $null
}

function Resolve-Conda {
    $condaCommand = Get-Command conda -ErrorAction SilentlyContinue
    if ($condaCommand) {
        return $condaCommand.Source
    }

    $candidates = @(
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        "$env:ProgramData\miniconda3\Scripts\conda.exe",
        "$env:ProgramData\anaconda3\Scripts\conda.exe"
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }
    throw "Could not find conda. Open an Anaconda/Miniconda-enabled terminal or add conda to PATH."
}

Write-Host "Audiveris path: $AudiverisPath"
Write-Host "PDF input dir: $PdfDir"
Write-Host "MusicXML output dir: $OutDir"
Write-Host "Conda env: $CondaEnv"

if (-not $SkipOmr) {
    if (-not (Test-Path $AudiverisPath)) {
        throw "Audiveris not found at: $AudiverisPath"
    }
    if (-not (Test-Path $PdfDir)) {
        throw "PDF directory not found: $PdfDir"
    }

    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    $listFile = Join-Path $OutDir "audiveris_inputs.txt"

    $pdfFiles = Get-ChildItem -Path $PdfDir -Recurse -File -Filter *.pdf
    if ($pdfFiles.Count -eq 0) {
        throw "No PDF files found in: $PdfDir"
    }

    # Audiveris @argfile expects raw paths (no wrapping quotes).
    # Keep ASCII to avoid UTF-8 BOM issues in PowerShell 5 argument files.
    $pdfFiles | Select-Object -ExpandProperty FullName | Set-Content -Path $listFile -Encoding ascii

    Write-Host "Running Audiveris on $($pdfFiles.Count) PDFs..."
    $audiverisArgs = @("-batch", "-transcribe", "-export", "-output", $OutDir)
    if ($PlainXml) {
        $audiverisArgs += @("-constant", "org.audiveris.omr.sheet.BookManager.useCompression=false")
    }
    $audiverisArgs += @("--", "@$listFile")

    $resolvedJava = Resolve-AudiverisJava -AudiverisExe $AudiverisPath
    if ($null -ne $resolvedJava) {
        $javaArgs = @(
            "-Djpackage.app-version=5.9.0",
            "--add-exports=java.desktop/sun.awt.image=ALL-UNNAMED",
            "--enable-native-access=ALL-UNNAMED",
            "--add-opens=java.base/java.nio=ALL-UNNAMED",
            "-Dfile.encoding=UTF-8",
            "-Xms512m",
            "-Xmx8G",
            "-cp",
            (Join-Path $resolvedJava.AppDir "*"),
            "Audiveris"
        ) + $audiverisArgs

        Write-Host "Using bundled Java launcher to run Audiveris synchronously..."
        & $resolvedJava.JavaExe @javaArgs
    }
    else {
        Write-Warning "Could not resolve bundled Java runtime. Falling back to Audiveris.exe."
        & $AudiverisPath @audiverisArgs
    }

    if ($LASTEXITCODE -ne 0) {
        throw "Audiveris failed with exit code $LASTEXITCODE"
    }

    $xmlAfterOmr = Get-MusicXmlFiles -Dir $OutDir
    if ($xmlAfterOmr.Count -eq 0) {
        $lastLog = Get-ChildItem -Path $OutDir -File -Filter *.log -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if ($null -ne $lastLog) {
            throw "Audiveris completed but no .xml/.musicxml/.mxl files were found. Check log: $($lastLog.FullName)"
        }
        throw "Audiveris completed but no .xml/.musicxml/.mxl files were found in $OutDir"
    }
}
else {
    Write-Host "Skipping OMR step (--SkipOmr)"
}

if (-not $SkipExtract) {
    $repoRoot = Split-Path -Parent $PSScriptRoot
    $extractScript = Join-Path $repoRoot "scripts\extract_time_signatures.py"
    if (-not (Test-Path $extractScript)) {
        throw "Extractor script missing: $extractScript"
    }

    $outCsv = Join-Path $OutDir "time_signatures_by_measure.csv"
    $condaExe = Resolve-Conda

    Write-Host "Extracting time signatures with music21..."
    $xmlFiles = Get-MusicXmlFiles -Dir $OutDir
    if ($xmlFiles.Count -eq 0) {
        throw "No MusicXML files found in $OutDir. Run without -SkipOmr first, or inspect Audiveris logs."
    }
    & $condaExe run -n $CondaEnv python $extractScript --xml-dir $OutDir --out-csv $outCsv
    if ($LASTEXITCODE -ne 0) {
        throw "Extraction failed with exit code $LASTEXITCODE"
    }

    Write-Host "Done. CSV written to: $outCsv"
}
else {
    Write-Host "Skipping extraction step (--SkipExtract)"
}
