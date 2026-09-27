[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$VsDevCmd = 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat'
$Nvcc = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe'
if (!(Test-Path $VsDevCmd)) { throw "VS developer environment not found: $VsDevCmd" }
if (!(Test-Path $Nvcc)) { throw "nvcc not found: $Nvcc" }
$BuildDir = Join-Path $Root 'build'
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
$Source = Join-Path $Root 'experiments\cuda_basics\vector_add\vector_add_starter.cu'
$Output = Join-Path $BuildDir 'vector_add.exe'
$command = 'call "{0}" -arch=x64 -host_arch=x64 && "{1}" -std=c++17 -O2 -arch=sm_75 --allow-unsupported-compiler "{2}" -o "{3}"' -f $VsDevCmd, $Nvcc, $Source, $Output
& cmd.exe /d /c $command
if ($LASTEXITCODE -ne 0) { throw "nvcc failed with exit code $LASTEXITCODE" }
Write-Output "Built: $Output"
