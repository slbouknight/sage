# 0009: Persist the pipeline cache to disk, and validate its header

**Status:** Accepted

**Context:** [ADR 0002](0002-slang-for-shaders.md) committed to a
`VkPipelineCache`, but an in-memory cache only helps within a single run.
Pipeline creation is a real compiler backend running at startup; persisting its
output is what removes that cost across runs. The blob is opaque and specific
to a driver version and GPU, and the spec makes feeding foreign data to
`vkCreatePipelineCache` undefined behaviour.

**Decision:** The cache is written to `$XDG_CACHE_HOME/sage/pipeline_cache.bin`
(falling back to `~/.cache`) at the end of `run()`, not in a destructor, so the
I/O has somewhere to report failure. On load, the first 32 bytes are copied out
as a `VkPipelineCacheHeaderVersionOne` and checked for header version, vendor
ID, device ID, and `pipelineCacheUUID` before the data is trusted. Every
failure path logs and continues with a cold compile.

**Consequences:** Driver updates invalidate the cache automatically, since
vendors change `pipelineCacheUUID` whenever compiler output changes. The size
guard before the `memcpy` is load-bearing — without it a short file is a heap
overread, not merely an invalid cache. Keeping the file out of the build tree
means `--clean` does not discard it and a machine-specific blob cannot be
committed by accident. Abnormal exits lose that session's compilations, which
is acceptable for a pure optimization.
