#!/usr/bin/env bash
# Export the PLABench changes to each model fork as a git patch.
#
# forks/ is nine submodules pointing at github.com/lyuweiorg. These patches make
# that indirection optional: clone the upstream repository, check out the pinned
# commit, apply the patch, and you have the same tree. Run this after changing
# any fork, and commit the result.
#
# The pinned upstream commits are read out of the table in forks/README.md so
# there is one place to update when a fork is rebased.
set -euo pipefail

cd "$(dirname "$0")/.."
OUT=forks/patches
mkdir -p "$OUT"
rm -f "$OUT"/*.patch

written=0
skipped=()
while IFS=$'\t' read -r fork upstream base; do
    [ -d "forks/$fork/.git" ] || { echo "forks/$fork is not checked out" >&2; exit 1; }
    if ! git -C "forks/$fork" cat-file -e "${base}^{commit}" 2>/dev/null; then
        echo "forks/$fork: pinned commit $base is not in the object store" >&2
        exit 1
    fi
    git -C "forks/$fork" format-patch --stdout "$base"..HEAD >"$OUT/$fork.patch"
    if [ -s "$OUT/$fork.patch" ]; then
        printf '%-18s %-28s %s..%s  %s\n' "$fork" "$upstream" "$base" \
            "$(git -C "forks/$fork" rev-parse --short HEAD)" \
            "$(wc -c <"$OUT/$fork.patch" | tr -d ' ') B"
        written=$((written + 1))
    else
        rm -f "$OUT/$fork.patch"
        skipped+=("$fork")
    fi
done < <(awk -F'|' '
    NF >= 5 && $2 ~ /`/ {
        fork = $2; gsub(/[` ]/, "", fork)
        up = $3; gsub(/^ +| +$/, "", up)
        split($4, f, " "); base = f[1]; gsub(/[` ]/, "", base)
        if (fork != "" && base ~ /^[0-9a-f]{7,40}$/) print fork "\t" up "\t" base
    }' forks/README.md)

echo
echo "$written patches in $OUT"
[ ${#skipped[@]} -eq 0 ] || echo "unchanged from upstream, no patch: ${skipped[*]}"
