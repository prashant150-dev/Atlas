#!/usr/bin/env bash
# ATLAS -> GitHub push helper.
# Reads creds from .atlas_secrets (gitignored), pushes current branch to your Atlas repo.
# The token is used in-memory for the push only; it is NOT written to .git/config.
#
#   bash push.sh             # commit nothing new, just push current HEAD
#   bash push.sh "message"   # stage all + commit with message, then push
set -euo pipefail
cd "$(dirname "$0")"

SEC=".atlas_secrets"
if [[ ! -f "$SEC" ]]; then
  echo "ERROR: $SEC not found. Copy .atlas_secrets.example -> .atlas_secrets and fill it." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$SEC"
: "${GITHUB_USER:?set GITHUB_USER in .atlas_secrets}"
: "${GITHUB_TOKEN:?set GITHUB_TOKEN in .atlas_secrets}"
: "${GITHUB_REPO:?set GITHUB_REPO in .atlas_secrets}"
BRANCH="${GITHUB_BRANCH:-main}"

# Optional commit if a message was passed as the first arg (not a flag like --force).
if [[ "${1:-}" != "" && "${1:0:1}" != "-" ]]; then
  git add -A
  git commit -m "$1" || echo "(nothing to commit)"
  shift
fi
# Anything left in "$@" are extra git-push flags (e.g. --force).

URL="https://${GITHUB_USER}:${GITHUB_TOKEN}@github.com/${GITHUB_USER}/${GITHUB_REPO}.git"
echo ">> pushing $(git rev-parse --short HEAD) -> ${GITHUB_USER}/${GITHUB_REPO} (${BRANCH})"
git push "$URL" "HEAD:${BRANCH}" "$@" 2>&1 | sed -E "s/${GITHUB_TOKEN}/***TOKEN***/g"
echo ">> done."
