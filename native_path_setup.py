import os
import sys


def _prepend_env_path(path):
    if not path or not os.path.isdir(path):
        return
    current = os.environ.get("PATH", "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    normalized = {
        os.path.normcase(os.path.normpath(entry))
        for entry in entries
    }
    target = os.path.normcase(os.path.normpath(path))
    if target not in normalized:
        os.environ["PATH"] = path + (os.pathsep + current if current else "")


def _prepend_sys_path(path):
    if not path or not os.path.isdir(path):
        return
    normalized = os.path.normcase(os.path.normpath(path))
    existing = {
        os.path.normcase(os.path.normpath(entry))
        for entry in sys.path
        if entry
    }
    if normalized not in existing:
        sys.path.insert(0, path)


def ensure_runtime_environment():
    repo_root = os.path.dirname(os.path.abspath(__file__))
    python_prefix = sys.prefix
    for candidate in [
        os.path.join(python_prefix, "Scripts"),
        os.path.join(python_prefix, "Library", "bin"),
        os.path.join(python_prefix, "bin"),
    ]:
        _prepend_env_path(candidate)

    mpl_config_dir = os.path.join(repo_root, ".tmp", "matplotlib")
    os.makedirs(mpl_config_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_config_dir)


def ensure_local_native_module_paths():
    ensure_runtime_environment()
    repo_root = os.path.dirname(os.path.abspath(__file__))
    module_roots = [
        os.path.join(repo_root, "submodules", "simple-knn"),
        os.path.join(repo_root, "submodules", "bvh"),
        os.path.join(repo_root, "rgss-rasterization"),
        os.path.join(repo_root, "svgss_rasterization"),
        os.path.join(repo_root, "external", "torch_scatter"),
        os.path.join(repo_root, "external", "nvdiffrast"),
    ]
    for module_root in module_roots:
        _prepend_sys_path(module_root)
