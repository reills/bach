# windows_runner.ps1 - Windows-native automated coding-agent task loop for bach-gen
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\windows_runner.ps1 [OPTIONS]
#
# Options:
#   --dry-run           Print what would happen; do not execute agent commands
#   --init              Initialize (or reset) state file from prompts.md
#   --prompt-file FILE  Path to prompt markdown file (default: prompts.md)
#   --state-file FILE   Path to persistent state TSV (default: .task_runner_state.tsv)
#   --task TASKID       Force-run exactly this task ID (bypasses dependency check)
#   --help              Show this help and exit
#
# Environment variables:
#   AGENT_MODEL      Default model for both agents: "claude" or "codex" (default: codex)
#   IMPLEMENT_MODEL  Override model for implement agent
#   REVIEW_MODEL     Override model for review agent
#   IMPLEMENT_CMD    Override implement command
#   REVIEW_CMD       Override review command
#   MAX_RETRIES      Max FAIL retries before a task is marked blocked (default: 3)
#   AUTO_COMMIT      Set to 1 to git-commit after each PASS (default: 1)
#   STOP_ON_BLOCKED  Set to 1 to exit when a task is blocked (default: 1)
#   DRY_RUN          Set to 1 to enable dry-run mode (default: 0)
#   LAST_TASK        Stop after this task completes (example: P24)
#   GIT_BASH_EXE     Optional path to Git Bash for running .sh helper scripts

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Show-Help {
  Get-Content -Path $PSCommandPath | Select-Object -First 28
}

function Write-Log([string]$Level, [string]$Message) {
  $timestamp = Get-Date -Format "HH:mm:ss"
  Write-Host ("[{0}] {1} {2}" -f $timestamp, $Level, $Message)
}

function Write-Info([string]$Message) { Write-Log "INFO " $Message }
function Write-Warn([string]$Message) { Write-Log "WARN " $Message }
function Write-Err([string]$Message) { Write-Log "ERROR" $Message }

function Die([string]$Message) {
  Write-Err $Message
  exit 1
}

function Parse-RunnerArgs([string[]]$CliArgs) {
  $parsed = [ordered]@{
    DryRun = $false
    InitMode = $false
    PromptFile = "prompts.md"
    StateFile = ".task_runner_state.tsv"
    ForceTask = ""
  }

  for ($i = 0; $i -lt $CliArgs.Count; $i++) {
    switch ($CliArgs[$i]) {
      "--dry-run" { $parsed.DryRun = $true }
      "-DryRun" { $parsed.DryRun = $true }
      "--init" { $parsed.InitMode = $true }
      "-Init" { $parsed.InitMode = $true }
      "--prompt-file" {
        $i++
        if ($i -ge $CliArgs.Count) { Die "Missing value for --prompt-file" }
        $parsed.PromptFile = $CliArgs[$i]
      }
      "-PromptFile" {
        $i++
        if ($i -ge $CliArgs.Count) { Die "Missing value for -PromptFile" }
        $parsed.PromptFile = $CliArgs[$i]
      }
      "--state-file" {
        $i++
        if ($i -ge $CliArgs.Count) { Die "Missing value for --state-file" }
        $parsed.StateFile = $CliArgs[$i]
      }
      "-StateFile" {
        $i++
        if ($i -ge $CliArgs.Count) { Die "Missing value for -StateFile" }
        $parsed.StateFile = $CliArgs[$i]
      }
      "--task" {
        $i++
        if ($i -ge $CliArgs.Count) { Die "Missing value for --task" }
        $parsed.ForceTask = $CliArgs[$i]
      }
      "-Task" {
        $i++
        if ($i -ge $CliArgs.Count) { Die "Missing value for -Task" }
        $parsed.ForceTask = $CliArgs[$i]
      }
      "--help" { Show-Help; exit 0 }
      "-Help" { Show-Help; exit 0 }
      "-h" { Show-Help; exit 0 }
      default { Die ("Unknown argument: {0}" -f $CliArgs[$i]) }
    }
  }

  return $parsed
}

function Get-EnvOrDefault([string]$Name, [string]$Default) {
  if ([string]::IsNullOrEmpty([Environment]::GetEnvironmentVariable($Name))) {
    return $Default
  }

  return [Environment]::GetEnvironmentVariable($Name)
}

function Get-IntEnvOrDefault([string]$Name, [int]$Default) {
  $value = [Environment]::GetEnvironmentVariable($Name)
  if ([string]::IsNullOrWhiteSpace($value)) {
    return $Default
  }

  $parsed = 0
  if (-not [int]::TryParse($value, [ref]$parsed)) {
    Die ("Environment variable {0} must be an integer, got '{1}'" -f $Name, $value)
  }

  return $parsed
}

function Get-TaskData([string]$PromptFile) {
  $lines = Get-Content -Path $PromptFile
  $tasks = New-Object System.Collections.Generic.List[object]
  $current = $null
  $inPromptBlock = $false

  foreach ($line in $lines) {
    if ($line -match '^### (P\d+) - (.+)$') {
      if ($null -ne $current) {
        $tasks.Add([pscustomobject]@{
          Id = $current.Id
          Title = $current.Title
          DependencyLine = $current.DependencyLine
          TestsLine = $current.TestsLine
          Prompt = ($current.PromptLines -join "`n")
        })
      }

      $current = [ordered]@{
        Id = $Matches[1]
        Title = $Matches[2]
        DependencyLine = ""
        TestsLine = ""
        PromptLines = New-Object System.Collections.Generic.List[string]
      }
      $inPromptBlock = $false
      continue
    }

    if ($null -eq $current) {
      continue
    }

    if ($line -match '^- Dependency: `(.*)`') {
      $current.DependencyLine = $Matches[1]
      continue
    }

    if ($line -match '^- Tests:') {
      $current.TestsLine = $line
      continue
    }

    if ($line -match '^```text\s*$') {
      $inPromptBlock = $true
      continue
    }

    if ($inPromptBlock -and $line -match '^```\s*$') {
      $inPromptBlock = $false
      continue
    }

    if ($inPromptBlock) {
      $current.PromptLines.Add($line)
    }
  }

  if ($null -ne $current) {
    $tasks.Add([pscustomobject]@{
      Id = $current.Id
      Title = $current.Title
      DependencyLine = $current.DependencyLine
      TestsLine = $current.TestsLine
      Prompt = ($current.PromptLines -join "`n")
    })
  }

  if ($tasks.Count -eq 0) {
    Die ("No tasks found in prompt file: {0}" -f $PromptFile)
  }

  return $tasks
}

function Get-TaskIds([object[]]$Tasks) {
  return @($Tasks | ForEach-Object { $_.Id })
}

function Get-TaskById([hashtable]$TaskMap, [string]$TaskId) {
  if (-not $TaskMap.ContainsKey($TaskId)) {
    Die ("Unknown task ID: {0}" -f $TaskId)
  }

  return $TaskMap[$TaskId]
}

function Get-TaskDeps([object]$Task) {
  if ([string]::IsNullOrWhiteSpace($Task.DependencyLine)) {
    Write-Warn ("No Dependency line found for {0}; treating as Independent" -f $Task.Id)
    return @()
  }

  if ($Task.DependencyLine -eq "Independent") {
    return @()
  }

  return @([regex]::Matches($Task.DependencyLine, 'P\d+') | ForEach-Object { $_.Value })
}

function Get-TaskTests([object]$Task) {
  if ([string]::IsNullOrWhiteSpace($Task.TestsLine)) {
    return @()
  }

  return @([regex]::Matches($Task.TestsLine, '`([^`]+)`') | ForEach-Object { $_.Groups[1].Value })
}

function New-StateEntry([string]$Status, [int]$Retries) {
  return [pscustomobject]@{
    Status = $Status
    Retries = $Retries
  }
}

function Write-StateFile([string]$StateFile, [string[]]$TaskIds, [hashtable]$StateMap) {
  $lines = foreach ($id in $TaskIds) {
    if (-not $StateMap.ContainsKey($id)) {
      Die ("State file is missing task: {0}" -f $id)
    }

    $entry = $StateMap[$id]
    "{0}`t{1}`t{2}" -f $id, $entry.Status, $entry.Retries
  }

  [System.IO.File]::WriteAllLines((Resolve-Path -LiteralPath ".").Path + "\" + $StateFile, $lines)
}

function Initialize-State([string]$StateFile, [string[]]$TaskIds, [bool]$InitMode) {
  if ((Test-Path -LiteralPath $StateFile) -and -not $InitMode) {
    return
  }

  if ((Test-Path -LiteralPath $StateFile) -and $InitMode) {
    Write-Warn "Reinitializing state file. Existing state backed up."
    Copy-Item -LiteralPath $StateFile -Destination ("{0}.bak.{1}" -f $StateFile, [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()) -Force
  }

  Write-Info ("Writing state file: {0}" -f $StateFile)
  $stateMap = @{}
  foreach ($id in $TaskIds) {
    $stateMap[$id] = New-StateEntry "pending" 0
  }
  Write-StateFile -StateFile $StateFile -TaskIds $TaskIds -StateMap $stateMap
}

function Load-State([string]$StateFile, [string[]]$TaskIds) {
  if (-not (Test-Path -LiteralPath $StateFile)) {
    Die ("State file missing: {0}  (run --init first)" -f $StateFile)
  }

  $stateMap = @{}
  foreach ($line in Get-Content -Path $StateFile) {
    if ([string]::IsNullOrWhiteSpace($line)) {
      continue
    }

    $parts = $line -split "`t"
    if ($parts.Count -lt 3) {
      Die ("Malformed state line: {0}" -f $line)
    }

    $retries = 0
    if (-not [int]::TryParse($parts[2], [ref]$retries)) {
      Die ("Invalid retry count in state line: {0}" -f $line)
    }

    $stateMap[$parts[0]] = New-StateEntry $parts[1] $retries
  }

  foreach ($id in $TaskIds) {
    if (-not $stateMap.ContainsKey($id)) {
      Die ("State file is missing task: {0}" -f $id)
    }
  }

  return $stateMap
}

function Set-TaskStatus([hashtable]$StateMap, [string]$StateFile, [string[]]$TaskIds, [string]$TaskId, [string]$Status) {
  $StateMap[$TaskId].Status = $Status
  Write-StateFile -StateFile $StateFile -TaskIds $TaskIds -StateMap $StateMap
}

function Increment-TaskRetries([hashtable]$StateMap, [string]$StateFile, [string[]]$TaskIds, [string]$TaskId) {
  $StateMap[$TaskId].Retries++
  Write-StateFile -StateFile $StateFile -TaskIds $TaskIds -StateMap $StateMap
}

function Find-EligibleTask([string[]]$TaskIds, [hashtable]$TaskMap, [hashtable]$StateMap) {
  foreach ($id in $TaskIds) {
    $status = $StateMap[$id].Status
    if ($status -ne "pending" -and $status -ne "in_progress") {
      continue
    }

    $allMet = $true
    foreach ($dep in @(Get-TaskDeps (Get-TaskById -TaskMap $TaskMap -TaskId $id))) {
      if ($StateMap[$dep].Status -ne "completed") {
        $allMet = $false
        break
      }
    }

    if ($allMet) {
      return $id
    }
  }

  return ""
}

function Any-Pending([hashtable]$StateMap) {
  return @($StateMap.Values | Where-Object { $_.Status -eq "pending" -or $_.Status -eq "in_progress" }).Count -gt 0
}

function Render-Template([string]$TemplateFile, [string]$TaskId, [string]$TaskTitle, [int]$Attempt) {
  $content = Get-Content -Path $TemplateFile -Raw
  $content = $content.Replace("{{TASK_ID}}", $TaskId)
  $content = $content.Replace("{{TASK_TITLE}}", $TaskTitle)
  $content = $content.Replace("{{ATTEMPT}}", [string]$Attempt)
  return $content
}

if (-not ("WinCommandLine" -as [type])) {
  Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class WinCommandLine {
  [DllImport("shell32.dll", SetLastError = true)]
  public static extern IntPtr CommandLineToArgvW(string lpCmdLine, out int pNumArgs);

  [DllImport("kernel32.dll")]
  public static extern IntPtr LocalFree(IntPtr hMem);
}
"@
}

function Split-CommandLine([string]$CommandLine) {
  if ([string]::IsNullOrWhiteSpace($CommandLine)) {
    Die "Command line must not be empty"
  }

  $argc = 0
  $argvPtr = [WinCommandLine]::CommandLineToArgvW($CommandLine, [ref]$argc)
  if ($argvPtr -eq [IntPtr]::Zero) {
    Die ("Unable to parse command line: {0}" -f $CommandLine)
  }

  try {
    $args = New-Object string[] $argc
    for ($i = 0; $i -lt $argc; $i++) {
      $argPtr = [System.Runtime.InteropServices.Marshal]::ReadIntPtr($argvPtr, $i * [IntPtr]::Size)
      $args[$i] = [System.Runtime.InteropServices.Marshal]::PtrToStringUni($argPtr)
    }
    return $args
  }
  finally {
    [void][WinCommandLine]::LocalFree($argvPtr)
  }
}

function Format-WindowsArgument([string]$Argument) {
  if ($null -eq $Argument) {
    return '""'
  }

  if ($Argument.Length -eq 0) {
    return '""'
  }

  if ($Argument -notmatch '[\s"]') {
    return $Argument
  }

  $builder = New-Object System.Text.StringBuilder
  [void]$builder.Append('"')
  $backslashCount = 0

  foreach ($char in $Argument.ToCharArray()) {
    if ($char -eq '\') {
      $backslashCount++
      continue
    }

    if ($char -eq '"') {
      [void]$builder.Append('\' * ($backslashCount * 2 + 1))
      [void]$builder.Append('"')
      $backslashCount = 0
      continue
    }

    if ($backslashCount -gt 0) {
      [void]$builder.Append('\' * $backslashCount)
      $backslashCount = 0
    }

    [void]$builder.Append($char)
  }

  if ($backslashCount -gt 0) {
    [void]$builder.Append('\' * ($backslashCount * 2))
  }

  [void]$builder.Append('"')
  return $builder.ToString()
}

function Invoke-ExternalCommand([string[]]$CommandParts, [string]$WorkingDirectory, [string]$StdinText = $null) {
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = $CommandParts[0]
  $psi.WorkingDirectory = $WorkingDirectory
  $psi.UseShellExecute = $false
  $psi.RedirectStandardInput = ($null -ne $StdinText)
  $psi.Arguments = (($CommandParts | Select-Object -Skip 1) | ForEach-Object { Format-WindowsArgument $_ }) -join " "

  $process = New-Object System.Diagnostics.Process
  $process.StartInfo = $psi
  [void]$process.Start()

  if ($null -ne $StdinText) {
    $process.StandardInput.Write($StdinText)
    $process.StandardInput.Close()
  }

  $process.WaitForExit()
  return $process.ExitCode
}

function Invoke-CommandLine([string]$CommandLine, [string]$WorkingDirectory, [string]$StdinText = $null) {
  $psi = New-Object System.Diagnostics.ProcessStartInfo
  $psi.FileName = "cmd.exe"
  $psi.WorkingDirectory = $WorkingDirectory
  $psi.UseShellExecute = $false
  $psi.RedirectStandardInput = ($null -ne $StdinText)
  $psi.Arguments = "/d /s /c `"$CommandLine`""

  $process = New-Object System.Diagnostics.Process
  $process.StartInfo = $psi
  [void]$process.Start()

  if ($null -ne $StdinText) {
    $process.StandardInput.Write($StdinText)
    $process.StandardInput.Close()
  }

  $process.WaitForExit()
  return $process.ExitCode
}

function Get-BashExecutable {
  $override = [Environment]::GetEnvironmentVariable("GIT_BASH_EXE")
  if (-not [string]::IsNullOrWhiteSpace($override) -and (Test-Path -LiteralPath $override)) {
    return $override
  }

  $git = Get-Command git.exe -ErrorAction SilentlyContinue
  if ($null -ne $git) {
    $gitCmdDir = Split-Path -Parent $git.Source
    $gitRoot = Split-Path -Parent $gitCmdDir
    $candidates = @(
      (Join-Path $gitRoot "bin\bash.exe"),
      (Join-Path $gitRoot "usr\bin\bash.exe")
    )

    foreach ($candidate in $candidates) {
      if (Test-Path -LiteralPath $candidate) {
        return $candidate
      }
    }
  }

  $fallbacks = @(
    "C:\Program Files\Git\bin\bash.exe",
    "C:\Program Files\Git\usr\bin\bash.exe",
    "C:\Program Files (x86)\Git\bin\bash.exe"
  )

  foreach ($candidate in $fallbacks) {
    if (Test-Path -LiteralPath $candidate) {
      return $candidate
    }
  }

  return $null
}

function Run-Agent {
  param(
    [string]$Mode,
    [string]$Prompt,
    [string]$RunnerDir,
    [bool]$DryRun,
    [string]$ImplementModel,
    [string]$ReviewModel,
    [string]$ImplementCommand,
    [string]$ReviewCommand,
    [string]$WorkingDirectory
  )

  $activePromptFile = Join-Path $RunnerDir ("{0}_active_prompt.txt" -f $Mode)
  [System.IO.File]::WriteAllText($activePromptFile, $Prompt + "`n")

  $commandText = if ($Mode -eq "implement") { $ImplementCommand } else { $ReviewCommand }
  $agentModel = if ($Mode -eq "implement") { $ImplementModel } else { $ReviewModel }

  if ($DryRun) {
    Write-Info ("[DRY-RUN] Would run: {0}" -f $commandText)
    Write-Info "[DRY-RUN] Prompt preview (first 5 lines):"
    Get-Content -Path $activePromptFile | Select-Object -First 5 | ForEach-Object { Write-Host ("  > {0}" -f $_) }
    return $true
  }

  Write-Info ("Running {0} agent ({1})..." -f $Mode, $agentModel)
  $exitCode = 0

  if ($agentModel -eq "codex") {
    $stdinText = Get-Content -Path $activePromptFile -Raw
    $exitCode = Invoke-CommandLine -CommandLine $commandText -WorkingDirectory $WorkingDirectory -StdinText $stdinText
  }
  else {
    $promptText = Get-Content -Path $activePromptFile -Raw
    $claudeCommand = $commandText + " " + (Format-WindowsArgument $promptText)
    $exitCode = Invoke-CommandLine -CommandLine $claudeCommand -WorkingDirectory $WorkingDirectory
  }

  return $exitCode -eq 0
}

function Parse-Verdict([string]$ReviewFile) {
  if (-not (Test-Path -LiteralPath $ReviewFile)) {
    return "UNKNOWN"
  }

  foreach ($line in Get-Content -Path $ReviewFile) {
    if ($line -match '^VERDICT:\s+(\S+)') {
      return $Matches[1]
    }
  }

  return "UNKNOWN"
}

function Parse-RemainingWork([string]$ReviewFile) {
  if (-not (Test-Path -LiteralPath $ReviewFile)) {
    return ""
  }

  $lines = Get-Content -Path $ReviewFile
  $found = $false
  $remaining = New-Object System.Collections.Generic.List[string]
  foreach ($line in $lines) {
    if ($found) {
      $remaining.Add($line)
      continue
    }

    if ($line -eq "REMAINING_WORK:") {
      $found = $true
    }
  }

  return ($remaining -join "`n")
}

function Archive-File([string]$Source, [string]$Destination) {
  Copy-Item -LiteralPath $Source -Destination $Destination -Force
  Write-Info ("Archived: {0} -> {1}" -f $Source, $Destination)
}

function Invoke-Git([string[]]$Args, [string]$WorkingDirectory) {
  $git = Get-Command git.exe -ErrorAction SilentlyContinue
  if ($null -eq $git) {
    $git = Get-Command git -ErrorAction SilentlyContinue
  }
  if ($null -eq $git) {
    Die "git was not found on PATH"
  }

  $exitCode = Invoke-ExternalCommand -CommandParts (@($git.Source) + $Args) -WorkingDirectory $WorkingDirectory
  if ($exitCode -ne 0) {
    Die ("git command failed: git {0}" -f ($Args -join " "))
  }
}

function Do-Commit {
  param(
    [string]$TaskId,
    [string]$TaskTitle,
    [int]$AutoCommit,
    [bool]$DryRun,
    [string]$WorkingDirectory
  )

  if ($AutoCommit -ne 1) {
    return
  }

  if ($DryRun) {
    Write-Info ("[DRY-RUN] Would commit: {0}: {1}" -f $TaskId, $TaskTitle)
    return
  }

  Write-Info ("Auto-committing for {0}" -f $TaskId)
  Invoke-Git -Args @("add", "-A") -WorkingDirectory $WorkingDirectory

  $message = "${TaskId}: $TaskTitle`n`nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
  Invoke-Git -Args @("commit", "-m", $message) -WorkingDirectory $WorkingDirectory
}

function Invoke-TargetedTests([string[]]$Tests, [string]$WorkingDirectory) {
  $bashExe = Get-BashExecutable
  if ($null -eq $bashExe) {
    Write-Warn "Git Bash was not found; skipping pre-flight targeted tests."
    return $false
  }

  $quotedTests = $Tests | ForEach-Object { '"{0}"' -f $_ }
  $testCmd = "bash docs/skills/python-test-env/scripts/run_tests.sh {0}" -f ($quotedTests -join " ")
  $exitCode = Invoke-ExternalCommand -CommandParts @($bashExe, "-lc", $testCmd) -WorkingDirectory $WorkingDirectory
  return $exitCode -eq 0
}

function Print-State([string[]]$TaskIds, [hashtable]$StateMap) {
  Write-Host ""
  Write-Host ("{0,-6}  {1,-12}  {2}" -f "Task", "Status", "Retries")
  Write-Host ("{0,-6}  {1,-12}  {2}" -f "------", "------------", "-------")
  foreach ($id in $TaskIds) {
    $entry = $StateMap[$id]
    Write-Host ("{0,-6}  {1,-12}  {2}" -f $id, $entry.Status, $entry.Retries)
  }
  Write-Host ""
}

function Run-Task {
  param(
    [string]$TaskId,
    [hashtable]$TaskMap,
    [string[]]$TaskIds,
    [hashtable]$StateMap,
    [string]$StateFile,
    [string]$RunnerDir,
    [string]$ArchiveDir,
    [string]$TodoFile,
    [string]$FinishedFile,
    [string]$ReviewFile,
    [int]$MaxRetries,
    [int]$AutoCommit,
    [bool]$DryRun,
    [string]$ImplementModel,
    [string]$ReviewModel,
    [string]$ImplementCommand,
    [string]$ReviewCommand,
    [string]$WorkingDirectory
  )

  $task = Get-TaskById -TaskMap $TaskMap -TaskId $TaskId
  $taskTitle = $task.Title
  $retries = $StateMap[$TaskId].Retries
  $attempt = $retries + 1
  $originalStatus = $StateMap[$TaskId].Status

  Write-Info "-----------------------------------------"
  Write-Info ("Task {0} - attempt {1} / max {2}" -f $TaskId, $attempt, $MaxRetries)
  Write-Info ("Title: {0}" -f $taskTitle)
  Write-Info "-----------------------------------------"

  if ($attempt -gt $MaxRetries) {
    Write-Warn ("Task {0} exceeded max retries ({1}). Marking blocked." -f $TaskId, $MaxRetries)
    Set-TaskStatus -StateMap $StateMap -StateFile $StateFile -TaskIds $TaskIds -TaskId $TaskId -Status "blocked"
    return 1
  }

  Set-TaskStatus -StateMap $StateMap -StateFile $StateFile -TaskIds $TaskIds -TaskId $TaskId -Status "in_progress"

  $taskPrompt = $task.Prompt
  if ([string]::IsNullOrWhiteSpace($taskPrompt)) {
    Die ("Could not extract prompt body for {0}" -f $TaskId)
  }

  $taskTests = @(Get-TaskTests $task)
  $todoLines = New-Object System.Collections.Generic.List[string]
  $todoLines.Add("# TODO - Active Task: $TaskId")
  $todoLines.Add("")
  $todoLines.Add("## $TaskId - $taskTitle")
  $todoLines.Add("")
  $todoLines.AddRange(($taskPrompt -split "`n"))
  if ($taskTests.Count -gt 0) {
    $todoLines.Add("")
    $todoLines.Add("## Test Command")
    $todoLines.Add("")
    $todoLines.Add("Run ONLY these targeted tests (do NOT run the full suite):")
    $todoLines.Add("")
    $todoLines.Add('```bash')
    $todoLines.Add(("bash docs/skills/python-test-env/scripts/run_tests.sh {0}" -f ($taskTests -join " ")))
    $todoLines.Add('```')
  }
  [System.IO.File]::WriteAllLines((Join-Path $WorkingDirectory $TodoFile), $todoLines)
  Write-Info ("Wrote {0}" -f $TodoFile)

  if ((-not $DryRun) -and $taskTests.Count -gt 0 -and $attempt -gt 1) {
    Write-Info "Pre-flight: running targeted tests to check if work is already done..."
    if (Invoke-TargetedTests -Tests $taskTests -WorkingDirectory $WorkingDirectory) {
      Write-Info ("Pre-flight PASS - tests already green. Marking {0} completed." -f $TaskId)
      Set-TaskStatus -StateMap $StateMap -StateFile $StateFile -TaskIds $TaskIds -TaskId $TaskId -Status "completed"
      Do-Commit -TaskId $TaskId -TaskTitle $taskTitle -AutoCommit $AutoCommit -DryRun $DryRun -WorkingDirectory $WorkingDirectory
      [System.IO.File]::WriteAllText((Join-Path $WorkingDirectory $TodoFile), "")
      return 0
    }

    Write-Info "Pre-flight: tests not passing yet. Proceeding with agent."
  }

  Remove-Item -LiteralPath $FinishedFile -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath $ReviewFile -Force -ErrorAction SilentlyContinue

  Write-Info "--- IMPLEMENT PASS ---"
  $implPrompt = Render-Template -TemplateFile (Join-Path $RunnerDir "implement_instructions.txt") -TaskId $TaskId -TaskTitle $taskTitle -Attempt $attempt
  if (-not (Run-Agent -Mode "implement" -Prompt $implPrompt -RunnerDir $RunnerDir -DryRun $DryRun -ImplementModel $ImplementModel -ReviewModel $ReviewModel -ImplementCommand $ImplementCommand -ReviewCommand $ReviewCommand -WorkingDirectory $WorkingDirectory)) {
    Write-Err ("Implement agent exited non-zero for {0}" -f $TaskId)
    return 1
  }

  if (-not $DryRun) {
    if (-not (Test-Path -LiteralPath $FinishedFile)) {
      Write-Err ("Implement agent did not write {0} for {1}" -f $FinishedFile, $TaskId)
      Write-Err "Re-run after the agent writes this file, or fix the agent command."
      return 1
    }

    Archive-File -Source $FinishedFile -Destination (Join-Path $ArchiveDir ("{0}-impl-{1}.md" -f $TaskId, $attempt.ToString("00")))
  }

  Write-Info "--- REVIEW PASS ---"
  $reviewPrompt = Render-Template -TemplateFile (Join-Path $RunnerDir "review_instructions.txt") -TaskId $TaskId -TaskTitle $taskTitle -Attempt $attempt
  if (-not (Run-Agent -Mode "review" -Prompt $reviewPrompt -RunnerDir $RunnerDir -DryRun $DryRun -ImplementModel $ImplementModel -ReviewModel $ReviewModel -ImplementCommand $ImplementCommand -ReviewCommand $ReviewCommand -WorkingDirectory $WorkingDirectory)) {
    Write-Err ("Review agent exited non-zero for {0}" -f $TaskId)
    return 1
  }

  if (-not $DryRun) {
    if (-not (Test-Path -LiteralPath $ReviewFile)) {
      Write-Err ("Review agent did not write {0}" -f $ReviewFile)
      return 1
    }

    Archive-File -Source $ReviewFile -Destination (Join-Path $ArchiveDir ("{0}-review-{1}.md" -f $TaskId, $attempt.ToString("00")))
  }

  if ($DryRun) {
    Write-Info "[DRY-RUN] Would parse verdict and advance state"
    Set-TaskStatus -StateMap $StateMap -StateFile $StateFile -TaskIds $TaskIds -TaskId $TaskId -Status $originalStatus
    return 0
  }

  $verdict = Parse-Verdict -ReviewFile $ReviewFile
  Write-Info ("Verdict for {0}: {1}" -f $TaskId, $verdict)

  switch ($verdict) {
    "PASS" {
      Write-Info ("PASS - marking {0} completed" -f $TaskId)
      Set-TaskStatus -StateMap $StateMap -StateFile $StateFile -TaskIds $TaskIds -TaskId $TaskId -Status "completed"
      Do-Commit -TaskId $TaskId -TaskTitle $taskTitle -AutoCommit $AutoCommit -DryRun $DryRun -WorkingDirectory $WorkingDirectory
      [System.IO.File]::WriteAllText((Join-Path $WorkingDirectory $TodoFile), "")
      return 0
    }
    "FAIL" {
      Write-Warn ("FAIL - preparing remediation for {0}" -f $TaskId)
      Increment-TaskRetries -StateMap $StateMap -StateFile $StateFile -TaskIds $TaskIds -TaskId $TaskId
      $newRetries = $StateMap[$TaskId].Retries

      if ($newRetries -ge $MaxRetries) {
        Write-Err ("Task {0} has hit max retries ({1}). Marking blocked." -f $TaskId, $MaxRetries)
        Set-TaskStatus -StateMap $StateMap -StateFile $StateFile -TaskIds $TaskIds -TaskId $TaskId -Status "blocked"
        return 1
      }

      $remaining = Parse-RemainingWork -ReviewFile $ReviewFile
      $remediation = @(
        "# TODO - Remediation: $TaskId (attempt $($newRetries + 1))",
        "",
        "## Original Task: $TaskId - $taskTitle",
        "",
        "This is a remediation run. The previous attempt received a FAIL verdict.",
        "Address ONLY the remaining work items listed below. Do not redo work that passed.",
        "",
        "## Remaining Work",
        "",
        $remaining,
        "",
        "## Original Task Prompt (reference only)",
        "",
        $taskPrompt
      )
      [System.IO.File]::WriteAllLines((Join-Path $WorkingDirectory $TodoFile), $remediation)
      Write-Info ("Wrote remediation {0} (retry {1} of {2})" -f $TodoFile, ($newRetries + 1), $MaxRetries)

      Set-TaskStatus -StateMap $StateMap -StateFile $StateFile -TaskIds $TaskIds -TaskId $TaskId -Status "pending"
      return 2
    }
    default {
      Write-Err ("Unexpected verdict '{0}' in {1}" -f $verdict, $ReviewFile)
      Write-Err "Expected first line: 'VERDICT: PASS' or 'VERDICT: FAIL'"
      return 1
    }
  }
}

$parsedArgs = Parse-RunnerArgs $args
$dryRun = $parsedArgs.DryRun -or (Get-IntEnvOrDefault -Name "DRY_RUN" -Default 0) -eq 1
$initMode = $parsedArgs.InitMode
$promptFile = $parsedArgs.PromptFile
$stateFile = $parsedArgs.StateFile
$forceTask = $parsedArgs.ForceTask

function Get-DefaultCommandForModel([string]$Model) {
  switch ($Model) {
    "claude" { return "claude --dangerously-skip-permissions -p" }
    "codex" { return "codex exec --full-auto -C . -" }
    default { Die ("Unknown model: {0} (expected 'claude' or 'codex')" -f $Model) }
  }
}

$agentModel = Get-EnvOrDefault -Name "AGENT_MODEL" -Default "codex"
$implementModel = Get-EnvOrDefault -Name "IMPLEMENT_MODEL" -Default $agentModel
$reviewModel = Get-EnvOrDefault -Name "REVIEW_MODEL" -Default $agentModel
$implementCommand = Get-EnvOrDefault -Name "IMPLEMENT_CMD" -Default (Get-DefaultCommandForModel -Model $implementModel)
$reviewCommand = Get-EnvOrDefault -Name "REVIEW_CMD" -Default (Get-DefaultCommandForModel -Model $reviewModel)

$maxRetries = Get-IntEnvOrDefault -Name "MAX_RETRIES" -Default 3
$autoCommit = Get-IntEnvOrDefault -Name "AUTO_COMMIT" -Default 1
$stopOnBlocked = Get-IntEnvOrDefault -Name "STOP_ON_BLOCKED" -Default 1
$lastTask = Get-EnvOrDefault -Name "LAST_TASK" -Default ""
$runnerDir = ".runner"
$archiveDir = "finished_prompt_summary"
$todoFile = "TODO.md"
$finishedFile = "finished.md"
$reviewFile = "review.md"
$workingDirectory = (Get-Location).Path

Write-Info "windows_runner.ps1 starting"
if ($dryRun) {
  Write-Info "(DRY-RUN mode active)"
}

if (-not (Test-Path -LiteralPath $promptFile)) {
  Die ("Prompt file not found: {0}" -f $promptFile)
}
if (-not (Test-Path -LiteralPath (Join-Path $runnerDir "implement_instructions.txt"))) {
  Die ("Missing {0} - run from repo root after setup" -f (Join-Path $runnerDir "implement_instructions.txt"))
}
if (-not (Test-Path -LiteralPath (Join-Path $runnerDir "review_instructions.txt"))) {
  Die ("Missing {0} - run from repo root after setup" -f (Join-Path $runnerDir "review_instructions.txt"))
}

New-Item -ItemType Directory -Path $archiveDir -Force | Out-Null
New-Item -ItemType Directory -Path $runnerDir -Force | Out-Null

$tasks = Get-TaskData -PromptFile $promptFile
$taskIds = Get-TaskIds -Tasks $tasks
$taskMap = @{}
foreach ($task in $tasks) {
  $taskMap[$task.Id] = $task
}

Initialize-State -StateFile $stateFile -TaskIds $taskIds -InitMode $initMode
$stateMap = Load-State -StateFile $stateFile -TaskIds $taskIds

if ($initMode) {
  Write-Info "State initialized:"
  Print-State -TaskIds $taskIds -StateMap $stateMap
  Write-Info "Run without --init to start the task loop."
  exit 0
}

while ($true) {
  $currentTask = ""

  if (-not [string]::IsNullOrWhiteSpace($forceTask)) {
    $currentTask = $forceTask
    Get-TaskById -TaskMap $taskMap -TaskId $currentTask | Out-Null
    if ($stateMap[$currentTask].Status -eq "completed") {
      Write-Info ("Task {0} is already completed. Marking pending for forced re-run." -f $currentTask)
      Set-TaskStatus -StateMap $stateMap -StateFile $stateFile -TaskIds $taskIds -TaskId $currentTask -Status "pending"
    }
  }
  else {
    $currentTask = Find-EligibleTask -TaskIds $taskIds -TaskMap $taskMap -StateMap $stateMap
  }

  if ([string]::IsNullOrWhiteSpace($currentTask)) {
    if (Any-Pending -StateMap $stateMap) {
      Write-Warn "No eligible tasks found, but pending tasks remain."
      Write-Warn "All remaining pending tasks have unmet (or blocked) dependencies."
      Write-Info "Current state:"
      Print-State -TaskIds $taskIds -StateMap $stateMap
      exit 1
    }

    Write-Info "All tasks completed. Workflow done!"
    Print-State -TaskIds $taskIds -StateMap $stateMap
    exit 0
  }

  $exitCode = Run-Task `
    -TaskId $currentTask `
    -TaskMap $taskMap `
    -TaskIds $taskIds `
    -StateMap $stateMap `
    -StateFile $stateFile `
    -RunnerDir $runnerDir `
    -ArchiveDir $archiveDir `
    -TodoFile $todoFile `
    -FinishedFile $finishedFile `
    -ReviewFile $reviewFile `
    -MaxRetries $maxRetries `
    -AutoCommit $autoCommit `
    -DryRun $dryRun `
    -ImplementModel $implementModel `
    -ReviewModel $reviewModel `
    -ImplementCommand $implementCommand `
    -ReviewCommand $reviewCommand `
    -WorkingDirectory $workingDirectory

  switch ($exitCode) {
    0 {
      Write-Info ("Task {0} PASSED." -f $currentTask)
      if (-not [string]::IsNullOrWhiteSpace($lastTask) -and $currentTask -eq $lastTask) {
        Write-Info ("Reached LAST_TASK={0}. Stopping." -f $lastTask)
        Print-State -TaskIds $taskIds -StateMap $stateMap
        exit 0
      }
    }
    2 {
      Write-Info ("Task {0} will be retried on next iteration." -f $currentTask)
    }
    default {
      if ($stateMap[$currentTask].Status -eq "blocked") {
        if ($stopOnBlocked -eq 1) {
          Write-Err ("Task {0} is blocked. STOP_ON_BLOCKED=1. Halting." -f $currentTask)
          Print-State -TaskIds $taskIds -StateMap $stateMap
          exit 1
        }

        Write-Warn ("Task {0} is blocked. Continuing to next eligible task." -f $currentTask)
      }
      else {
        Write-Err ("Task {0} failed with exit code {1}. Runner stopping." -f $currentTask, $exitCode)
        Print-State -TaskIds $taskIds -StateMap $stateMap
        exit 1
      }
    }
  }

  if (-not [string]::IsNullOrWhiteSpace($forceTask) -or $dryRun) {
    if ($dryRun) {
      Write-Info "[DRY-RUN] Showing next eligible task only. Re-run without --dry-run to execute."
      Print-State -TaskIds $taskIds -StateMap $stateMap
    }
    break
  }
}
