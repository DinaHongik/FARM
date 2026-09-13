#!/bin/bash
# Store a HuggingFace token and verify it can reach the gated embeddinggemma repo.
# Server-side for the same reason as doctor.sh: PowerShell mangles embedded quotes.
#
#   set_token.sh hf_xxxxxxxxxxxx

TOKEN="${1:?usage: set_token.sh <hf_token>}"
mkdir -p ~/.cache/huggingface
printf '%s' "$TOKEN" > ~/.cache/huggingface/token
chmod 600 ~/.cache/huggingface/token
echo "token stored in ~/.cache/huggingface/token"

echo "verifying access to google/embeddinggemma-300m ..."
CODE=$(curl -s -o /dev/null -w '%{http_code}' \
  -H "Authorization: Bearer $TOKEN" \
  https://huggingface.co/api/models/google/embeddinggemma-300m)

case "$CODE" in
  200) echo "ACCESS OK (HTTP 200) - training can proceed" ;;
  401) echo "HTTP 401 - token is invalid or expired" ;;
  403) echo "HTTP 403 - token is valid but the licence has not been accepted."
       echo "           Visit https://huggingface.co/google/embeddinggemma-300m and accept it, then retry." ;;
  *)   echo "HTTP $CODE - unexpected; check network access from the server" ;;
esac
