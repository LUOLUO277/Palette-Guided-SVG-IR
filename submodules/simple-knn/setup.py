#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import sys

cxx_compiler_flags = []
nvcc_flags = ["-allow-unsupported-compiler"]
include_dirs = []
cuda_target_include = os.path.join(sys.prefix, "Library", "include", "targets", "x64")
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

    for version in ("14.36.32532", "14.43.34808"):
        root = os.path.join(
            "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC",
            version,
        )
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

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")

setup(
    name="simple_knn",
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=[
            "spatial.cu", 
            "simple_knn.cu",
            "ext.cpp"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            extra_compile_args={"nvcc": nvcc_flags, "cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
