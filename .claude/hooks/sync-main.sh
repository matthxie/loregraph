#!/usr/bin/env bash
# SessionStart hook: keep local `main` in sync with origin/main, and prune
# local branches whose PR has merged.
#
# Safe by construction — it only ever *fast-forwards* and *safe-deletes*
# (git branch -d, which refuses anything not fully merged). It never merges,
# rebases, resets, force-deletes, or discards work, and it exits 0 no matter
# what so it can never block a session from starting.
#
# Why this exists: merging a PR on GitHub advances origin/main but not your
# local main checkout, and leaves the merged local branch behind. Without
# this, new branches get cut from a stale base (conflicts) and dead branches
# pile up. There is no "PR merged" hook (that's a remote event), so this runs
# at session start — the first session after a merge cleans everything up.
set -u

# Run inside the project; bail quietly if we can't or it isn't a git repo.
cd "${CLAUDE_PROJECT_DIR:-$PWD}" 2>/dev/null || exit 0
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0

# Refresh remote refs and clear deleted remote branches. Bail on any
# network/auth failure so a flaky connection never delays startup.
git fetch --prune --quiet origin 2>/dev/null || exit 0

branch="$(git symbolic-ref --short -q HEAD || true)"
if [ "$branch" = "main" ]; then
  # On main: fast-forward only, and only when the tree is clean so we never
  # touch uncommitted work.
  if [ -z "$(git status --porcelain 2>/dev/null)" ]; then
    git merge --ff-only --quiet origin/main 2>/dev/null || true
  fi
else
  # Off main (e.g. a feature branch or worktree): advance the local main ref
  # to origin/main without checking it out. The refspec form fast-forwards
  # only and no-ops harmlessly if main is checked out in another worktree.
  git fetch --quiet origin main:main 2>/dev/null || true
fi

# Prune merged branches: delete any local branch whose upstream is gone (the
# state after a PR merges with GitHub's auto-delete-branch on). `-d` is the
# safety net — it refuses any branch not fully merged, and we never force, so
# unmerged work is never lost. The current branch and main are skipped.
git for-each-ref --format='%(refname:short) %(upstream:track)' refs/heads 2>/dev/null \
  | while read -r ref track; do
      [ "$ref" = "main" ] && continue
      [ "$ref" = "$branch" ] && continue
      [ "$track" = "[gone]" ] && git branch -d "$ref" >/dev/null 2>&1
    done

exit 0
