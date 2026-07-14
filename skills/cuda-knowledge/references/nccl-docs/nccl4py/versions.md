# Versions

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/nccl4py/versions.html

---

# Versions[](#versions "Permalink to this heading")

NCCL4Py exposes top-level helpers to inspect the installed NCCL stack: `nccl4py` itself plus the native libraries `libnccl.so` and `libnccl_ep.so`.
    
    
    import nccl
    nccl.show_versions()      # human-readable block to stdout
    v = nccl.get_version()    # programmatic snapshot
    

## show_versions[](#show-versions "Permalink to this heading")

nccl.show_versions() → None[](#nccl.show_versions "Permalink to this definition")
    

Print a summary of the installed NCCL stack to stdout.

For each component, reports the release version, the CUDA toolkit it was built with, and (for native libraries) the path of the loaded `.so`.

## get_version[](#get-version "Permalink to this heading")

nccl.get_version() → [VersionInfo](#nccl.VersionInfo "nccl._show_versions.VersionInfo")[](#nccl.get_version "Permalink to this definition")
    

Return a structured snapshot of NCCL stack versions.

Returns:
    

[`VersionInfo`](#nccl.VersionInfo "nccl.VersionInfo") with nccl4py + libnccl + libnccl_ep versions, CUDA build variants, and loaded `.so` paths.

## VersionInfo[](#versioninfo "Permalink to this heading")

_class _nccl.VersionInfo(_nccl4py : [Version](https://packaging.pypa.io/en/stable/version.html#packaging.version.Version "\(in Packaging v26.2\)")_, _nccl : [LibraryInfo](#nccl.LibraryInfo "nccl._show_versions.LibraryInfo") | None_, _nccl_ep : [LibraryInfo](#nccl.LibraryInfo "nccl._show_versions.LibraryInfo") | None_)[](#nccl.VersionInfo "Permalink to this definition")
    

Bases: `object`

Aggregate version snapshot of the NCCL stack.

nccl4py _: [Version](https://packaging.pypa.io/en/stable/version.html#packaging.version.Version "\(in Packaging v26.2\)")_[](#nccl.VersionInfo.nccl4py "Permalink to this definition")
    

nccl4py package version.

nccl _: [LibraryInfo](#nccl.LibraryInfo "nccl._show_versions.LibraryInfo") | None_[](#nccl.VersionInfo.nccl "Permalink to this definition")
    

Version/CUDA-variant/path of the `libnccl.so` nccl4py is using, or None when it cannot be loaded.

nccl_ep _: [LibraryInfo](#nccl.LibraryInfo "nccl._show_versions.LibraryInfo") | None_[](#nccl.VersionInfo.nccl_ep "Permalink to this definition")
    

Version/CUDA-variant/path of the `libnccl_ep.so` nccl4py is using, or None when it cannot be loaded.

## LibraryInfo[](#libraryinfo "Permalink to this heading")

_class _nccl.LibraryInfo(_version : [Version](https://packaging.pypa.io/en/stable/version.html#packaging.version.Version "\(in Packaging v26.2\)")_, _cuda_variant : [Version](https://packaging.pypa.io/en/stable/version.html#packaging.version.Version "\(in Packaging v26.2\)") | None_, _path : Path | None_)[](#nccl.LibraryInfo "Permalink to this definition")
    

Bases: `object`

Version, CUDA build variant, and loaded-path info for a shared library.

version _: [Version](https://packaging.pypa.io/en/stable/version.html#packaging.version.Version "\(in Packaging v26.2\)")_[](#nccl.LibraryInfo.version "Permalink to this definition")
    

Library release version (e.g. `2.30.0`).

cuda_variant _: [Version](https://packaging.pypa.io/en/stable/version.html#packaging.version.Version "\(in Packaging v26.2\)") | None_[](#nccl.LibraryInfo.cuda_variant "Permalink to this definition")
    

CUDA toolkit major.minor the library was built with (e.g. `12.9`), or None if it could not be read from the library.

path _: Path | None_[](#nccl.LibraryInfo.path "Permalink to this definition")
    

Path of the loaded `libnccl.so`, or None if it cannot be determined.