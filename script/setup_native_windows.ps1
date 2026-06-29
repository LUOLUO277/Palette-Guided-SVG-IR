param(
    [string]$PythonExe = "C:\Users\c2483\anaconda3\envs\svgir37p\python.exe",
    [string]$CondaExe = "C:\Users\c2483\anaconda3\Scripts\conda.exe",
    [string]$EnvPrefix = "C:\Users\c2483\anaconda3\envs\svgir37p",
    [string]$VCToolsVersion = "14.36.32532",
    [switch]$SkipCondaCuda,
    [switch]$SkipPipDeps,
    [switch]$SkipBuild,
    [switch]$SkipNvdiffrast
)

$ErrorActionPreference = "Stop"

function Assert-PathExists {
    param([string]$PathValue, [string]$Label)
    if (-not (Test-Path -LiteralPath $PathValue)) {
        throw "$Label not found: $PathValue"
    }
}

function Import-VsDevEnvironment {
    param([string]$ToolsetVersion)
    $vsDevCmd = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
    Assert-PathExists $vsDevCmd "VsDevCmd"

    $args = @("-arch=x64", "-host_arch=x64")
    if ($ToolsetVersion) {
        $args += "-vcvars_ver=$ToolsetVersion"
    }
    $cmd = "`"$vsDevCmd`" $($args -join ' ') >nul && set"
    $envDump = & cmd.exe /d /s /c $cmd
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to initialize Visual Studio build environment."
    }

    foreach ($line in $envDump) {
        $idx = $line.IndexOf("=")
        if ($idx -lt 1) {
            continue
        }
        $name = $line.Substring(0, $idx)
        $value = $line.Substring($idx + 1)
        Set-Item -Path "Env:$name" -Value $value
    }

    $msvcRoots = @()
    if ($ToolsetVersion) {
        $msvcRoots += "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\$ToolsetVersion"
    }
    $msvcRoots += "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.43.34808"

    $pathPrefixes = @()
    $libPrefixes = @()
    $windowsSdkBinX64 = "C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64"
    if (Test-Path -LiteralPath $windowsSdkBinX64) {
        $pathPrefixes += $windowsSdkBinX64
    }
    foreach ($vcRoot in $msvcRoots) {
        if (-not (Test-Path -LiteralPath $vcRoot)) {
            continue
        }
        $compilerBin = Join-Path $vcRoot "bin\HostX64\x64"
        if (Test-Path -LiteralPath $compilerBin) {
            $pathPrefixes += $compilerBin
        }
        foreach ($relativeLibDir in @("lib\x64", "atlmfc\lib\x64")) {
            $candidate = Join-Path $vcRoot $relativeLibDir
            if (Test-Path -LiteralPath $candidate) {
                $libPrefixes += $candidate
            }
        }
    }
    if ($pathPrefixes.Count -gt 0) {
        $env:PATH = (($pathPrefixes + @($env:PATH)) | Where-Object { $_ } | Select-Object -Unique) -join ";"
    }
    if ($libPrefixes.Count -gt 0) {
        $env:LIB = (($libPrefixes + @($env:LIB)) | Where-Object { $_ } | Select-Object -Unique) -join ";"
    }
}

function Find-CondaCudaHome {
    param([string]$Prefix)
    $candidates = @(
        (Join-Path $Prefix "Library"),
        (Join-Path $Prefix ""),
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8",
        "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6"
    )

    foreach ($candidate in $candidates) {
        if (-not $candidate) { continue }
        $nvccA = Join-Path $candidate "bin\nvcc.exe"
        $nvccB = Join-Path $candidate "nvcc.exe"
        if ((Test-Path -LiteralPath $nvccA) -or (Test-Path -LiteralPath $nvccB)) {
            return $candidate
        }
    }
    return $null
}

function Invoke-Step {
    param([string]$Label, [scriptblock]$Action)
    Write-Host "==> $Label"
    & $Action
}

Assert-PathExists $PythonExe "Python"
Assert-PathExists $CondaExe "Conda"
Assert-PathExists $EnvPrefix "Conda env"

Import-VsDevEnvironment -ToolsetVersion $VCToolsVersion
$env:PATH = "$(Join-Path $EnvPrefix 'Scripts');$(Join-Path $EnvPrefix 'Library\bin');$(Join-Path $EnvPrefix 'bin');$env:PATH"
$env:DISTUTILS_USE_SDK = "1"
$env:MSSdk = "1"
$mplConfigDir = Join-Path $PWD ".tmp\matplotlib"
New-Item -ItemType Directory -Force -Path $mplConfigDir | Out-Null
$env:MPLCONFIGDIR = $mplConfigDir

if (-not $SkipCondaCuda) {
    Invoke-Step "Installing CUDA compiler toolchain into the conda env" {
        & $CondaExe install -y -p $EnvPrefix -c nvidia `
            cuda-nvcc=11.8.89 `
            cuda-nvcc_win-64=11.8.0 `
            cuda-cudart=11.8.89 `
            cuda-cudart-dev=11.8.89
        if ($LASTEXITCODE -ne 0) {
            throw "conda install for CUDA toolchain failed."
        }
    }
}

$cudaHome = Find-CondaCudaHome -Prefix $EnvPrefix
if (-not $cudaHome) {
    throw "CUDA toolkit not found after setup. Expected nvcc inside the conda env or a system CUDA install."
}

$env:CUDA_HOME = $cudaHome
$env:CUDA_PATH = $cudaHome
$env:TORCH_CUDA_ARCH_LIST = "8.6"
$env:PATH = "$(Join-Path $cudaHome 'bin');$(Join-Path $cudaHome 'libnvvp');$env:PATH"

if (-not $SkipPipDeps) {
    Invoke-Step "Installing Python dependencies for the native stack" {
        & $PythonExe -m pip install --upgrade pip setuptools wheel
        if ($LASTEXITCODE -ne 0) { throw "Failed to upgrade pip toolchain." }

        & $PythonExe -m pip install slangtorch==1.2.1
        if ($LASTEXITCODE -ne 0) { throw "Failed to install slangtorch." }

        & $PythonExe -m pip install OpenEXR==1.3.9 Imath==0.0.2 pyexr==0.3.10 --no-build-isolation
        if ($LASTEXITCODE -ne 0) { throw "Failed to install pyexr/OpenEXR." }
    }
}

if (-not $SkipBuild) {
    Invoke-Step "Building project native extensions" {
        $torchScatterDir = Join-Path $PWD "external\torch_scatter"
        Assert-PathExists $torchScatterDir "Vendored torch_scatter"
        $env:FORCE_CUDA = "1"
        Push-Location $torchScatterDir
        & $PythonExe .\setup.py build_ext --inplace
        $exitCode = $LASTEXITCODE
        Pop-Location
        if ($exitCode -ne 0) { throw "Failed to build external\\torch_scatter inplace." }

        Push-Location .\submodules\bvh
        & $PythonExe .\setup.py build_ext --inplace
        $exitCode = $LASTEXITCODE
        Pop-Location
        if ($exitCode -ne 0) { throw "Failed to build submodules\\bvh inplace." }

        Push-Location .\submodules\simple-knn
        & $PythonExe .\setup.py build_ext --inplace
        $exitCode = $LASTEXITCODE
        Pop-Location
        if ($exitCode -ne 0) { throw "Failed to build submodules\\simple-knn inplace." }

        Push-Location .\rgss-rasterization
        & $PythonExe .\setup.py build_ext --inplace
        $exitCode = $LASTEXITCODE
        Pop-Location
        if ($exitCode -ne 0) { throw "Failed to build rgss-rasterization inplace." }

        Push-Location .\svgss_rasterization
        & $PythonExe .\setup.py build_ext --inplace
        $exitCode = $LASTEXITCODE
        Pop-Location
        if ($exitCode -ne 0) { throw "Failed to build svgss_rasterization inplace." }

        if (-not $SkipNvdiffrast) {
            $nvdiffrastDir = Join-Path $PWD "external\nvdiffrast"
            Assert-PathExists $nvdiffrastDir "Vendored nvdiffrast"
            Push-Location $nvdiffrastDir
            & $PythonExe .\setup.py build_ext --inplace
            $exitCode = $LASTEXITCODE
            Pop-Location
            if ($exitCode -ne 0) { throw "Failed to build external\\nvdiffrast inplace." }
        }
    }
}

Write-Host "==> Native environment setup complete"
Write-Host "CUDA_HOME=$env:CUDA_HOME"
