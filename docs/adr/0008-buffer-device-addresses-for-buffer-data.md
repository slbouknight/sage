# 0008: Buffer device addresses for buffer data, descriptors for images

**Status:** Accepted

**Context:** With [ADR 0007](0007-one-bindless-descriptor-set.md) in place there
are two ways for a shader to reach a buffer: index it out of the bindless
descriptor array, or dereference a 64-bit device address. Using descriptors for
everything means a descriptor write per buffer and an index to track; using
addresses for everything is impossible, since images need format and layout
metadata that a raw pointer cannot carry.

**Decision:** Plain buffer data is reached by buffer device address, passed in
push constants and dereferenced as a typed pointer in Slang. Images — and
buffers that genuinely benefit from descriptor indirection — go through the
bindless set. `Buffer` only calls `vkGetBufferDeviceAddress` when created with
`VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT`, and buffers reached by address
carry no descriptor-related usage flags at all.

**Consequences:** Vertex data needs no descriptor write, no index, and no
descriptor-array slot; an address can also be stored inside another buffer,
which is what makes GPU-driven draws practical later. The cost is that an
address is unvalidated — a stale or wrong one faults the GPU rather than
tripping a layer. The push-constant struct is duplicated in Slang and C++, so
`static_assert(offsetof(...))` pins the C++ side to the offsets the SPIR-V
declares.
