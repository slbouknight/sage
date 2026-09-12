# 0030: Authoring a scene — context menu, primitives, and lights as nodes

**Status:** Accepted

**Context:** v1.0 could open a glTF and look at it. It could not build anything.
Everything in a scene came from a file, lights were two `Application` members
behind a panel, and a single loaded model floated in a void with nothing to
receive its shadow — which is why M9's helmet shot has no shadow in it at all.

**Decision:** Three pieces, in the order they unblock each other.

**A right-click menu**, which collides with the camera: the right button already
means "fly". The press now only *arms* look mode and four pixels of movement
commit to it, so a release before that is a click. Committing on press — what
this used to do — makes the two undecidable, because `GLFW_CURSOR_DISABLED`
unbinds the pointer the instant the button goes down and every position after
it is a free-running virtual coordinate rather than a screen one. The distance
actually moved is gone by the time you want to ask about it.

That decision lives in `DragGesture`, free of GLFW, because there is no way to
drive a real cursor from a test and this is the part that would be wrong. The
threshold is a radius rather than a per-axis test: a 3,3 nudge is 4.24 px and
counts as a drag.

New objects land where the cursor ray meets the ground plane. The alternatives
were worse for the same result: there is no retained geometry to raycast against
on the CPU, and the depth buffer is `storeOp DONT_CARE` and not `SAMPLED`, so
there is nothing to read back either. Plane intersection is arithmetic, needs no
GPU involvement, and is exactly right for placing things on a floor.

**Primitives** — plane, cube, sphere, cone, cylinder — generated on the CPU into
the same `Vertex` layout glTF loads into. Kept pure so they could be tested,
which paid immediately: the winding test caught the cylinder's sides wound
backwards, which would have rendered the whole side inside out with back faces
culled, and 64 zero-area triangles at the sphere's poles where the quad against
the pole ring collapses to a point.

They are generated at unit size and scaled into the scene, because a fixed size
cannot serve both a chess piece and a street lamp — the sample models alone span
a factor of 36. A ground plane inflates the scene diagonal that subsequent
solids are measured against, which is why the solid fraction is a third rather
than the half it started as.

**Lights as scene nodes.** `SceneNode` gained a `SceneLight`, deliberately not a
`gpu::Light`: half of that struct — position and direction — is derived from the
node's transform rather than authored, and storing both would be two sources of
truth with the gizmo moving only one. Position is the world translation;
direction is the node's **-Y axis**, so an unrotated light points down and the
rotate gizmo aims it.

Each light carries a small icon mesh as a child, built from the primitive
generator, with an emissive material so it reads as a source rather than a grey
blob someone forgot to delete. Without it a light has no geometry, so object-id
picking cannot hit it and the outline has nothing to draw — it would be
reachable only from the hierarchy, exactly when leaving the 3D view is least
wanted. `editor_only` keeps icons out of the shadow pass, since a marker
standing for a light must not cast one, and out of captures, which are meant to
be the render rather than the tool.

**Consequences:** Three defaults were wrong, and all three looked fine.

The default light shone along `-X, -Y, -Z` while the camera starts at `+X, +Y,
+Z` and framing returns it there — straight down the view axis, so every shadow
fell directly behind the subject where the subject hid it. It reads exactly like
shadows being broken. A difference mask put 0.16% of the frame in shadow, all of
it a sliver at the edge of the helmet; angling the light across the view instead
took it to 0.59% and made it plainly visible. This was M9's default too — the
hero shot used a hand-picked direction, so it never showed.

`scene_bounds()` counted the light icons. With the helmet loaded it reported a
top of 2.07 where the scene ended at 0.9, so the shadow frustum stretched to
cover a marker floating in empty air and spent its resolution there. Editor
furniture is now excluded, which makes shadows slightly sharper in every scene
for a reason unrelated to anything M10 set out to do.

And two default lights were created at startup — one by the constructor, one by
the `clear_scene` that a command-line load runs on its way in. Only one
survived; the other was an upload thrown away a line later.

`clear_scene` re-adds a default light, because a light is a node and clearing
the graph removes it. Without that, loading a second model would drop the scene
into darkness.
