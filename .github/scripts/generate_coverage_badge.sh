#!/usr/bin/env bash
set -euo pipefail

# This script assumes it's run from the repository root and that
# coverage-percent.txt exists or will be created by the previous step.

if [ ! -f coverage-percent.txt ]; then
  echo "coverage-percent.txt missing; will use 'unknown'"
  echo unknown > coverage-percent.txt
fi

PERCENT=$(cat coverage-percent.txt)

if ! echo "$PERCENT" | grep -E '^[0-9]+(\.[0-9]+)?$' >/dev/null; then
  echo "coverage-percent.txt malformed or unknown; using 'unknown'"
  PERCENT=unknown
fi

# Treat zero or negative as a skip (don't update badge)
if [ "$PERCENT" != "unknown" ]; then
  if awk "BEGIN{exit !($PERCENT>0)}"; then
    :
  else
    echo "coverage percent is zero or negative; skipping badge update"
    exit 0
  fi
fi

mkdir -p docs

if [ "$PERCENT" = "unknown" ]; then
  # Unknown -> red
  COLOR=e05d44
  DISPLAY_TEXT='unknown'
else
  # Apply thresholds: green >=85, yellow >=75 and <85, orange >0, red for 0
  if awk "BEGIN{exit !($PERCENT>=85)}"; then
    COLOR=4c1
  elif awk "BEGIN{exit !($PERCENT>=75)}"; then
    COLOR=f7df1e
  elif awk "BEGIN{exit !($PERCENT>0)}"; then
    COLOR=f77f00
  else
    COLOR=e05d44
  fi
  DISPLAY_TEXT="$(printf '%.1f%%' "$PERCENT")"
fi

cat > docs/coverage-badge.svg <<EOF
<svg xmlns="http://www.w3.org/2000/svg" width="120" height="20">
  <rect width="120" height="20" fill="#555"/>
  <rect x="60" width="60" height="20" fill="#${COLOR}"/>
  <g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="11">
    <text x="30" y="14">coverage</text>
    <text x="90" y="14">${DISPLAY_TEXT}</text>
  </g>
</svg>
EOF

# Git operations: create/update fixed branch and open or update PR
git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"

# Fetch remote state and decide branch strategy. Some repos prevent pushes
# to a fixed branch if there are pending PRs or branch protections. In that
# case, use a timestamped branch to avoid failures and still create a PR.
git fetch origin --prune
REMOTE_BRANCH_EXISTS=0
if git ls-remote --heads origin coverage-badge-update | grep -q refs/heads/coverage-badge-update; then
  REMOTE_BRANCH_EXISTS=1
fi

if [ "$REMOTE_BRANCH_EXISTS" -eq 0 ]; then
  BRANCH="coverage-badge-update"
  git checkout -B "$BRANCH"
else
  # Create a unique branch name to avoid protected-branch or pending-PR issues
  TS=$(date -u +%Y%m%d%H%M%S)
  BRANCH="coverage-badge-update-${TS}"
  git checkout -b "$BRANCH"
fi

# Add only the badge file and percent file to avoid committing unrelated changes
git add docs/coverage-badge.svg coverage-percent.txt || true
if git diff --staged --quiet; then
  echo "No changes to commit"
  echo "Attempting to push branch (if it exists remotely)"
  if ! git push -u origin "$BRANCH"; then
    echo "git push failed (no staged changes). Printing git status and remote info for debugging."
    git status --porcelain --untracked-files=all || true
    git branch -vv || true
    git remote show origin || true
    echo "Continuing to PR creation step (push may have failed)."
  fi
else
  echo "Committing staged badge changes"
  if ! git commit -m "chore(ci): update coverage badge"; then
    echo "git commit failed. Showing git status and staged diffs for debugging:" >&2
    git status --porcelain --untracked-files=all || true
    echo "--- Staged diff ---"
    git diff --staged || true
    echo "--- End staged diff ---"
    # Do not exit; continue to attempt push and PR creation
  fi

  # Try pushing with a few retries to handle transient remote errors
  MAX_RETRIES=3
  n=0
  until [ $n -ge $MAX_RETRIES ]
  do
    echo "git push attempt $((n+1))"
    if git push -u origin "$BRANCH"; then
      echo "git push succeeded"
      break
    else
      echo "git push failed on attempt $((n+1))" >&2
      n=$((n+1))
      sleep $((n*2))
    fi
  done
  if [ $n -ge $MAX_RETRIES ]; then
    echo "git push failed after $MAX_RETRIES attempts; printing remote diagnostics" >&2
    git remote show origin || true
    echo "Continuing to PR creation step (push may have failed)."
  fi
fi

# If push failed after retries, create a timestamped branch and push it so PR creation
# can proceed from a unique branch (avoids non-fast-forward rejection on protected refs).
if [ ${n:-0} -ge $MAX_RETRIES ]; then
  TS=$(date -u +%Y%m%d%H%M%S)
  NEWBRANCH="${BRANCH}-${TS}"
  echo "Attempting fallback: creating new branch $NEWBRANCH and pushing it"
  # Create the new branch at current HEAD and push it
  git checkout -B "$NEWBRANCH"
  if git push -u origin "$NEWBRANCH"; then
    echo "Fallback push succeeded to $NEWBRANCH"
    BRANCH="$NEWBRANCH"
  else
    echo "Fallback push to $NEWBRANCH also failed. PR creation may fail." >&2
  fi
fi

# Use API to find existing PR for the branch
OWNER=${GITHUB_REPOSITORY%%/*}
EXISTING_PR_JSON=$(curl -s -H "Authorization: token $GITHUB_TOKEN" -H "Accept: application/vnd.github+json" "https://api.github.com/repos/$GITHUB_REPOSITORY/pulls?state=open&head=$OWNER:$BRANCH")
PR_NUMBER=$(echo "$EXISTING_PR_JSON" | grep -Po '"number":\s*\K[0-9]+' | head -n1 || true)

if [ -n "$PR_NUMBER" ]; then
  echo "Found existing PR #$PR_NUMBER - updating PR body"
  PAYLOAD=$(printf '{"title":"chore(ci): update coverage badge","body":"Automated coverage badge update: %s"}' "$DISPLAY_TEXT")
  curl -s -X PATCH -H "Authorization: token $GITHUB_TOKEN" -H "Accept: application/vnd.github+json" \
    https://api.github.com/repos/$GITHUB_REPOSITORY/pulls/$PR_NUMBER \
    -d "$PAYLOAD" || echo "PR update may have failed"
else
  PAYLOAD=$(printf '{"title":"chore(ci): update coverage badge","head":"%s","base":"main","body":"Automated coverage badge update: %s"}' "$BRANCH" "$DISPLAY_TEXT")
  curl -s -X POST -H "Authorization: token $GITHUB_TOKEN" -H "Accept: application/vnd.github+json" \
    https://api.github.com/repos/$GITHUB_REPOSITORY/pulls \
    -d "$PAYLOAD" || echo "PR creation may have failed"
fi

echo "Badge script completed"
