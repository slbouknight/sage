# 0010: Synchronous staging uploads with queue family ownership transfer

**Status:** Accepted

**Context:** Device-local memory is not host-visible, so M2's
`Buffer::write` — a `memcpy` into a persistently mapped allocation — does not
extend to the geometry buffer. Uploads need a staging buffer, a copy recorded on
the transfer queue, and synchronisation proving the copy finished before the
graphics queue reads the data. The selected GPU exposes a transfer-only family
(family 1) distinct from graphics (family 0), so `VK_SHARING_MODE_EXCLUSIVE`
buffers additionally require an ownership transfer.

**Decision:** `Uploader` performs one fully synchronous upload per call: stage,
copy plus release barrier on the transfer queue, fence wait, acquire barrier on
the graphics queue, fence wait, then destroy the staging buffer. Callers supply
the acquire barrier's destination scope, so vertex regions acquire with
`VERTEX_SHADER`/`SHADER_STORAGE_READ` and index regions with
`INDEX_INPUT`/`INDEX_READ`. `VK_SHARING_MODE_CONCURRENT` was rejected: it would
remove the ownership dance but can disable driver-side compression, and it would
mean the portable path never runs on this hardware.

**Consequences:** Startup-only. Every upload stalls both queues and allocates a
fresh staging buffer, which is the wrong shape for streaming but the right
amount of machinery for static geometry. The two failure modes this ordering
prevents are both silent — freeing staging while the copy is in flight, and
releasing ownership without acquiring it, which many drivers tolerate and
others corrupt. Streaming would need a persistent staging ring and semaphores
rather than fence waits; that is a rewrite of `Uploader`'s internals, not of its
interface.
