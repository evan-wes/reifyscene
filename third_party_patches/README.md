# Patches to third-party code

Every edit reifyscene makes to a third-party checkout lives here, one numbered `.patch` file per
change, and every installer applies them through `scripts/lib/patch_series.sh`. If you want to
know what we have changed in someone else's code, this directory is the complete answer.

```
third_party_patches/<tool>/NNNN-short-slug.patch
```

## Why patch files rather than `sed` in the installer

That is what this replaced. The problem is not tidiness:

- **`sed -i` exits 0 when it matches nothing.** An upstream that drifted would silently ship
  unpatched code, failing much later and somewhere else. Every patch therefore needed a
  hand-written sentinel check (`require_patched`, anchor asserts) to compensate — verification
  that had to be remembered each time. `git apply` fails when the context no longer matches, so
  that check is now structural rather than optional.
- **A diff is reviewable as a diff.** A `sed` expression with escaped regex is not.
- **Regenerating after a pin bump** is `git -C <checkout> diff`, not re-deriving a substitution.
- **The reason lives with the change.** Each file opens with free text — what, why, which pin,
  and what breaks without it — which `git apply` ignores.

## Applying

```bash
source "${ROOT_DIR}/scripts/lib/patch_series.sh"
apply_patch_series "${SOM_DIR}" "${ROOT_DIR}/third_party_patches/shape-of-motion" "Shape of Motion"
```

Per patch, in filename order: already applied → skip; applies cleanly → apply; neither → stop the
install and say so. Re-running an installer is a no-op. A partially-patched tree is an error, not
something to work around.

## Adding one

1. Make the edit in the checkout by hand and confirm it works.
2. `git -C third_party/<tool> diff -- <files>` → the diff.
3. Save it as `third_party_patches/<tool>/NNNN-slug.patch` with a header saying what it does, the
   pinned commit, and what breaks without it.
4. Verify against a clean tree — `tests/test_third_party_patches.py` does this automatically.

Do not reference the installer inside the patched code; point at the patch file instead. The
whole point is one place to look.

## After bumping a pin

The patch will stop applying, loudly, which is the intended behaviour — an upstream that moved may
have fixed the problem or changed its shape. Check whether the patch is still needed before
regenerating it.

## Current inventory

| Tool | Pin | Patches |
|---|---|---|
| `shape-of-motion` | `579753e1` | torch device index; gsplat radii shape |
| `hloc` | `v1.4` (`80ccb7ee`) | pycolmap ≥ 3.10 API drift |
| `vggt-long` | `c160869d` | dinov2 CWD path; Hugging Face CLI rename |
| `concept-graphs` | `93277a02` | ConceptGraphs-Detect imports made optional |

Four other checkouts — `co-tracker`, `MoGe`, `sam2` and `vggt` — need no patches and have no
directory here.
