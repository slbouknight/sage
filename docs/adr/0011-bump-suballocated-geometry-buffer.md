# 0011: One bump-suballocated geometry buffer, no free

**Status:** Accepted

**Context:** A buffer per mesh means an allocation, a device address, and an
index-buffer binding per mesh, and it scatters geometry across memory. The
milestone calls for a single suballocated vertex/index buffer instead. That
raises two questions: how space is handed out, and how one buffer serves two
access paths.

**Decision:** One device-local `VkBuffer` created with
`INDEX_BUFFER | SHADER_DEVICE_ADDRESS | TRANSFER_DST`, carved up by a bump
allocator — a monotonically advancing offset with no free. Vertex regions align
to 16 (conservative for BDA-addressed structs); index regions align to 4, which
`vkCmdBindIndexBuffer` requires. `add_mesh` returns a `MeshView` holding a
device address for the vertices and a byte offset for the indices, because
vertices are dereferenced as pointers while `vkCmdBindIndexBuffer` is
fixed-function and takes a buffer plus offset. Vertex layout stays the caller's
concern; the registry only moves bytes.

**Consequences:** Meshes cannot be freed or replaced, and capacity is fixed at
construction — exceeding it asserts rather than growing. For M3's static
geometry that is the whole requirement, and a free list would be machinery with
no user. Loading a scene larger than the capacity, or unloading anything, forces
a real allocator; the `MeshView` interface is what keeps that change local to
the registry. Because device addresses are ordinary integers, a mesh's address
is `base + offset`, so one address query at construction serves every mesh.
