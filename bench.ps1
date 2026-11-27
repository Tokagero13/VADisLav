# Benchmark: wall time, CPU time, peak memory and speed of vad on the given files.
#   pwsh bench.ps1 a.webm b.mp4            the files as they are
#   pwsh bench.ps1 a.webm -Hour            also one-hour versions of the first file in 5 formats (needs ffmpeg)
[CmdletBinding(PositionalBinding = $false)]
param(
    [Parameter(Mandatory, ValueFromRemainingArguments)][string[]]$Files,
    [switch]$Hour,
    [int]$Runs = 5,
    [string]$Vad = (Join-Path $PSScriptRoot 'dist/vad.exe'),
    [string]$WorkDir = (Join-Path ([IO.Path]::GetTempPath()) 'vad-bench')
)
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force $WorkDir | Out-Null
$out = Join-Path $WorkDir 'out.wav'

if ($Hour) {
    $wav = Join-Path $WorkDir 'hour.wav' # the first file repeated up to one hour, 48 kHz stereo
    ffmpeg -v error -y -stream_loop -1 -i $Files[0] -t 3600 -ar 48000 -ac 2 $wav
    $formats = [ordered]@{
        'hour.mp4'  = '-f', 'lavfi', '-i', 'testsrc=s=640x360:r=5', '-shortest', '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac'
        'hour.webm' = '-c:a', 'libopus', '-b:a', '32k', '-ac', '1'
        'hour.mp3'  = '-q:a', '5'
        'hour.flac' = @()
    }
    foreach ($name in $formats.Keys) {
        $file = Join-Path $WorkDir $name
        $extra = $formats[$name]
        ffmpeg -v error -y -i $wav @extra $file
        $Files += $file
    }
    $Files += $wav
}

$rows = foreach ($file in $Files) {
    $best = $null
    $peakAll = 0
    foreach ($run in 1..$Runs) {
        $info = [Diagnostics.ProcessStartInfo]::new($Vad)
        $info.ArgumentList.Add($file); $info.ArgumentList.Add('-o'); $info.ArgumentList.Add($out)
        $info.RedirectStandardError = $true; $info.RedirectStandardOutput = $true
        $clock = [Diagnostics.Stopwatch]::StartNew()
        $p = [Diagnostics.Process]::Start($info)
        $peak = 0
        # memory can only be read while the process lives: sampled every 2 ms, so tiny runs read low
        while (-not $p.WaitForExit(2)) { try { $p.Refresh(); $peak = [Math]::Max($peak, $p.PeakWorkingSet64) } catch {} }
        $clock.Stop()
        $summary = $p.StandardError.ReadToEnd()
        if ($summary -notmatch 'kept ([\d.]+) s of ([\d.]+) s in [\d.]+ s \(decode ([\d.]+), detect ([\d.]+), write ([\d.]+)\)') { throw "$file`: $summary" }
        $r = [pscustomobject]@{
            File = (Split-Path $file -Leaf) -replace '^(.{28}).+(.{8})$', '$1…$2'; 'MB' = [Math]::Round((Get-Item $file).Length / 1MB, 1)
            'Audio s' = [double]$Matches[2]; 'Kept s' = [double]$Matches[1]
            'Wall s' = [Math]::Round($clock.Elapsed.TotalSeconds, 3); 'CPU s' = [Math]::Round($p.TotalProcessorTime.TotalSeconds, 3)
            'Decode s' = [double]$Matches[3]; 'Detect s' = [double]$Matches[4]; 'Write s' = [double]$Matches[5]
            'Peak MB' = [Math]::Round($peak / 1MB); 'x realtime' = [Math]::Round([double]$Matches[2] / $clock.Elapsed.TotalSeconds)
        }
        $peakAll = [Math]::Max($peakAll, $r.'Peak MB')
        if (-not $best -or $r.'Wall s' -lt $best.'Wall s') { $best = $r }
    }
    $best.'Peak MB' = $peakAll # the highest reading of all runs: sampling can only read too low
    $best
}
"vad $('{0:N1}' -f ((Get-Item $Vad).Length / 1MB)) MB, best of $Runs runs, $((Get-CimInstance Win32_Processor).Name), one thread"
$rows | Format-Table -Property * -AutoSize | Out-String -Width 200
