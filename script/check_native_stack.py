import importlib
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from native_path_setup import ensure_local_native_module_paths


ensure_local_native_module_paths()


def check_module(name):
    try:
        mod = importlib.import_module(name)
        path = getattr(mod, "__file__", None)
        version = getattr(mod, "__version__", None)
        print(f"[OK] {name} path={path} version={version}")
        return True
    except Exception as exc:
        print(f"[ERR] {name} {type(exc).__name__}: {exc}")
        return False


def check_cuda_toolchain():
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc_path = None
    if not cuda_home:
        prefix = os.path.dirname(sys.executable)
        for candidate in (
            os.path.join(prefix, "Library"),
            prefix,
        ):
            nvcc = os.path.join(candidate, "bin", "nvcc.exe")
            if os.path.exists(nvcc):
                cuda_home = candidate
                nvcc_path = nvcc
                os.environ.setdefault("CUDA_HOME", cuda_home)
                os.environ.setdefault("CUDA_PATH", cuda_home)
                os.environ["PATH"] = os.path.join(candidate, "bin") + os.pathsep + os.environ["PATH"]
                break
    elif os.path.exists(os.path.join(cuda_home, "bin", "nvcc.exe")):
        nvcc_path = os.path.join(cuda_home, "bin", "nvcc.exe")
    print(f"CUDA_HOME={cuda_home}")
    try:
        result = subprocess.run(
            [nvcc_path or "nvcc", "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(result.stdout.strip())
        return True
    except Exception as exc:
        print(f"[ERR] nvcc {type(exc).__name__}: {exc}")
        return False


def main():
    print(f"python={sys.version}")
    try:
        import torch

        print(f"torch={torch.__version__}")
        print(f"torch_cuda={torch.version.cuda}")
        print(f"cuda_available={torch.cuda.is_available()}")
        print(f"device_count={torch.cuda.device_count()}")
    except Exception as exc:
        print(f"[ERR] torch {type(exc).__name__}: {exc}")
        return 1

    ok = True
    ok &= check_cuda_toolchain()
    for name in [
        "pyexr",
        "torch_scatter",
        "slangtorch",
        "nvdiffrast.torch",
        "simple_knn._C",
        "bvh_tracing._C",
        "rgss_rasterization._C",
        "svgss_rasterization._C",
    ]:
        ok &= check_module(name)

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
