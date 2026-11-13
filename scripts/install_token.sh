#!/usr/bin/env bash
# Store the GitHub token once, use it everywhere.
#
#   bash scripts/install_token.sh
#
# Writes ~/.config/codeplan/env (mode 600) and sources it from ~/.bashrc, so every
# new terminal has GITHUB_TOKEN set. The collector also reads that file directly,
# so it works even without a shell restart.
set -euo pipefail

DIR="$HOME/.config/codeplan"
FILE="$DIR/env"
RC="$HOME/.bashrc"

read -rsp "GitHub token (paste, nothing is shown): " TOKEN
echo
TOKEN="$(printf '%s' "$TOKEN" | tr -d '[:space:]')"

if [[ -z "$TOKEN" ]]; then
  echo "No token entered." >&2
  exit 1
fi
case "$TOKEN" in
  ghp_*|github_pat_*) ;;
  *)
    echo "Unexpected prefix. Classic tokens start with ghp_, fine-grained with github_pat_." >&2
    exit 1
    ;;
esac
if [[ "$TOKEN" == ghp_ghp_* || "$TOKEN" == github_pat_github_pat_* ]]; then
  echo "Doubled prefix (ghp_ghp_...). Paste only what GitHub shows." >&2
  exit 1
fi

status="$(curl -s -o /dev/null -w '%{http_code}' \
  -H "Authorization: Bearer $TOKEN" https://api.github.com/user || true)"
if [[ "$status" != "200" ]]; then
  echo "GitHub answered HTTP $status for this token. Not saved." >&2
  echo "Create a new classic token without scopes and try again." >&2
  exit 1
fi

mkdir -p "$DIR"
umask 077
printf 'export GITHUB_TOKEN=%s\n' "$TOKEN" > "$FILE"
chmod 600 "$FILE"

LINE='[ -f "$HOME/.config/codeplan/env" ] && . "$HOME/.config/codeplan/env"'
if ! grep -qxF "$LINE" "$RC" 2>/dev/null; then
  printf '\n%s\n' "$LINE" >> "$RC"
  echo "Added the source line to $RC"
fi

login="$(curl -s -H "Authorization: Bearer $TOKEN" https://api.github.com/user \
  | python3 -c 'import json,sys; print(json.load(sys.stdin).get("login","?"))')"
echo "Token works (user: $login), stored in $FILE (mode 600)."
echo "Open a new terminal, or run: source $FILE"
