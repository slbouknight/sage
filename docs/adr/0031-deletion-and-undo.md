# 0031: Deleting by tombstone, and the undo stack that falls out of it

**Status:** Accepted

**Context:** Once a scene could be built it had to be unbuilt. `SceneGraph` had
no removal at all — M6 anticipated one ("deleting a node bumps its generation so
stale handles compare unequal") but it was never written. Undo was asked for at
the same time, and the two turned out to be one decision rather than two.

**Decision:** Deletion **tombstones** rather than erases. A node is marked dead
and stays in the array.

Compaction was the obvious alternative and is wrong here. A `NodeHandle` carries
an index, so erasing a slot slides everything past it down and silently
repoints every surviving handle: delete node 3 and a handle to node 5 now names
node 6, with generations still matching, so nothing detects it. Three other
things are indexed the same way — an object id is `index + 1`, a selection flag
is indexed by node, and a parent reference is a handle — and all three would
have to be remapped in step. Marking keeps every one of them correct for free,
and keeps parents ahead of their children with no re-sort.

Generations are deliberately *not* bumped on delete. A tombstoned slot is never
reused before `clear()`, so nothing can be confused with it, and leaving the
handle resolvable is what lets undo hand the same handle back. `find()` returns
null for a deleted node so every existing caller treats it as gone; `is_deleted`
separates "deleted" from "never valid", which is a distinction only undo needs.

The cost is that the array only grows and the deleted node's geometry stays
resident — the registries bump-allocate and free nothing below a full reset, per
[ADR 0011](0011-bump-suballocated-geometry-buffer.md). The read-out counts
`live_size()` rather than slots so the divergence is visible.

**That cost is also what makes undo cheap**, which was the genuine surprise.
Undoing an addition looked like the hard case: it should mean freeing geometry
the registries cannot free. With tombstones it is just tombstoning, and undoing
a deletion is un-tombstoning. Nothing is uploaded again, nothing is rewound. The
limitation that looks like a weakness is the one that makes the feature nearly
free.

Commands are closure pairs rather than a hierarchy of command types: each
operation here is a few captured values and two calls into the graph, and a
class per verb would be more machinery than the thing it describes. The stack is
a line, not a tree — a new edit after an undo discards what was ahead — and is
capped so a long session cannot grow it without limit.

Drags are coalesced. The gizmo and the Properties drags both write a transform
every frame they are held, so the value is recorded when the gesture starts and
one command is pushed when it ends; otherwise a single drag would leave a
hundred entries and Ctrl+Z would rewind a frame at a time. A gesture that ends
where it began pushes nothing, so undo never appears to do nothing.

**Consequences:** `Clear scene` is not undoable and drops the history. It
rewinds the registries, so the geometry is genuinely gone and no handle on the
stack still resolves. By extension a *replacing* load is not undoable either,
since it clears on the way in; an additive one is. That asymmetry is real and
deliberate — a Ctrl+Z that silently did nothing would be worse than one that is
honestly unavailable.

One bug is worth recording because it was invisible until exercised: the default
light was pushing a command. A fresh scene therefore started with one entry on
the stack, and the very first Ctrl+Z deleted the only light and dropped the
viewport into darkness, undoing something the user never did. The default light
is setup, not an edit, and is now created with recording suppressed — a fresh
scene reports an empty stack.

The scene-graph half is unit tested, including the case the whole design exists
for: remove node 0, then confirm handles to nodes 1 and 2 still resolve to the
right nodes. The stack itself is verified by exercising a full
add/transform/delete/undo-to-empty/redo-to-end sequence and reading the live
count, cursor and a transformed value at each step; it lives on `Application`,
which needs a device, so it is not reachable from a unit test.
