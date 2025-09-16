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
BRANCH="coverage-badge-update"
git checkout -B "$BRANCH"
git add docs/coverage-badge.svg coverage-percent.txt || true
if git diff --staged --quiet; then
  echo "No changes to commit"
  git push -u origin "$BRANCH" || true
else
  git commit -m "chore(ci): update coverage badge"
  git push -u origin "$BRANCH"
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
