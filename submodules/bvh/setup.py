import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_src_path = os.path.dirname(os.path.abspath(__file__))
nvcc_flags = ["-O3", "--expt-extended-lambda", "-allow-unsupported-compiler"]
cxx_flags = ["/O2"] if os.name == "nt" else ["-O3"]
cuda_target_include = os.path.join(sys.prefix, "Library", "include", "targets", "x64")
include_dirs = [
    os.path.join(_src_path, 'include'),
]
if os.path.exists(cuda_target_include):
    include_dirs.append(cuda_target_include)


def prepend_env_path(var_name, entries):
    current = os.environ.get(var_name, "")
    parts = [entry for entry in entries if entry and os.path.isdir(entry)]
    if current:
        parts.append(current)

    deduped = []
    seen = set()
    for part in parts:
        normalized = os.path.normcase(os.path.normpath(part))
        if normalized in seen:
            continue
        deduped.append(part)
        seen.add(normalized)
    if deduped:
        os.environ[var_name] = ";".join(deduped)


def ensure_windows_toolchain_env():
    if os.name != "nt":
        return

    path_entries = [
        os.path.join(sys.prefix, "Scripts"),
        os.path.join(sys.prefix, "Library", "bin"),
        os.path.join(sys.prefix, "bin"),
        "C:\\Program Files (x86)\\Windows Kits\\10\\bin\\10.0.22621.0\\x64",
    ]
    lib_entries = []
    for version in ("14.36.32532", "14.43.34808"):
        root = os.path.join(
            "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC",
            version,
        )
        path_entries.append(os.path.join(root, "bin", "HostX64", "x64"))
        for relative in ("lib\\x64", "atlmfc\\lib\\x64"):
            full_path = os.path.join(root, relative)
            if os.path.isdir(full_path):
                lib_entries.append(full_path)

    prepend_env_path("PATH", path_entries)
    prepend_env_path("LIB", lib_entries)


def collect_msvc_library_dirs():
    candidates = []
    env_lib = os.environ.get("LIB", "")
    for entry in env_lib.split(";"):
        if entry and os.path.isdir(entry):
            candidates.append(entry)

    toolset_roots = [
        os.path.join(
            "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC",
            version,
        )
        for version in ("14.36.32532", "14.43.34808")
    ]
    for root in toolset_roots:
        for relative in ("lib\\x64", "atlmfc\\lib\\x64"):
            full_path = os.path.join(root, relative)
            if os.path.isdir(full_path):
                candidates.append(full_path)

    deduped = []
    seen = set()
    for path in candidates:
        normalized = os.path.normcase(os.path.normpath(path))
        if normalized in seen:
            continue
        deduped.append(path)
        seen.add(normalized)
    return deduped


ensure_windows_toolchain_env()
library_dirs = collect_msvc_library_dirs()

setup(
    name='bvh_tracing',
    description='CUDA RayTracer with BVH acceleration for 3DGS',
    ext_modules=[
        CUDAExtension(
            name='bvh_tracing._C',
            sources=[os.path.join(_src_path, 'src', f) for f in [
                'bvh.cu',
                'trace.cu',
                'construct.cu',
                'bindings.cpp',
            ]],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            extra_compile_args={
                "nvcc": nvcc_flags,
                "cxx": cxx_flags}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension,
    },
)
