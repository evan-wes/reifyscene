# Apply this repo's patches to a third-party checkout.
#
# Every edit reifyscene makes to third-party code lives in third_party_patches/<tool>/ as a
# numbered .patch file, and every installer applies them through this one function. Before this
# existed the same job was done by `sed -i` and inline Python string replacement scattered across
# two installers, each with its own hand-written verification -- because `sed -i` exits 0 when it
# matches nothing, so an upstream that drifted would silently ship unpatched code. `git apply`
# fails when the context no longer matches, which makes that verification free rather than
# something each patch has to remember to write.
#
# Usage:
#   source "${ROOT_DIR}/scripts/lib/patch_series.sh"
#   apply_patch_series "${SOM_DIR}" "${ROOT_DIR}/third_party_patches/shape-of-motion" "Shape of Motion"
#
# To regenerate a patch after bumping a pin: check the new commit out, re-apply the edit by hand,
# then `git -C <checkout> diff -- <file>` and paste it under the existing header.

apply_patch_series() {
  local checkout="$1" patch_dir="$2" label="${3:-$(basename "${checkout}")}"
  local patch applied=0 skipped=0

  # Ask git rather than testing for a .git directory: in a worktree (and in a submodule) .git is
  # a FILE, and a -d test rejects a perfectly good checkout.
  if ! git -C "${checkout}" rev-parse --git-dir >/dev/null 2>&1; then
    echo "PATCH FAILED: ${checkout} is not a git checkout; cannot apply ${label} patches." >&2
    return 1
  fi
  if [[ ! -d "${patch_dir}" ]]; then
    echo "PATCH FAILED: no patch directory at ${patch_dir}" >&2
    return 1
  fi

  shopt -s nullglob
  local patches=("${patch_dir}"/*.patch)
  shopt -u nullglob
  if [[ ${#patches[@]} -eq 0 ]]; then
    echo "  ${label}: no patches to apply"
    return 0
  fi

  for patch in "${patches[@]}"; do
    # Already applied? Re-running an installer must be a no-op, and a checkout that a previous
    # run patched is the normal case, not an error.
    if git -C "${checkout}" apply --reverse --check "${patch}" >/dev/null 2>&1; then
      skipped=$((skipped + 1))
      continue
    fi
    if git -C "${checkout}" apply --check "${patch}" >/dev/null 2>&1; then
      git -C "${checkout}" apply "${patch}"
      applied=$((applied + 1))
      continue
    fi
    # Neither applies nor is applied: upstream drifted, or the tree is partly patched. Both are
    # states that must stop the install rather than proceed on code we have not verified.
    echo "PATCH FAILED: $(basename "${patch}") does not apply to ${checkout}" >&2
    echo "  pinned HEAD: $(git -C "${checkout}" rev-parse --short HEAD 2>/dev/null || echo unknown)" >&2
    echo "  The patch neither applies cleanly nor is already applied. Either the pin moved and" >&2
    echo "  the patch needs regenerating, or the checkout is partially patched." >&2
    echo "  Inspect:  git -C ${checkout} diff" >&2
    echo "  Reset:    git -C ${checkout} checkout -- . && re-run this installer" >&2
    return 1
  done

  echo "  ${label}: ${applied} patch(es) applied, ${skipped} already present"
  return 0
}
