#!/usr/bin/env bash
set -euo pipefail

# Bump version across all workspace crates.
# Usage: ./scripts/bump-version.sh <new_version>
#
# Guards:
#   1. Version must be valid semver (X.Y.Z).
#   2. CHANGELOG.md must contain an entry for the new version.
#   3. After bumping, every crate Cargo.toml must match the workspace version.

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <version>" >&2
    exit 1
fi

VERSION="$1"

# Validate semver format
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: Invalid version format: $VERSION (expected X.Y.Z)" >&2
    exit 1
fi

# Check CHANGELOG entry exists
if ! grep -q "## \[$VERSION\]" CHANGELOG.md && ! grep -q "## \[Unreleased\]" CHANGELOG.md; then
    echo "Error: CHANGELOG.md has no entry for [$VERSION] or [Unreleased]." >&2
    echo "Add a changelog entry before bumping." >&2
    exit 1
fi

echo "Bumping all crates to version $VERSION"

# Update workspace version in root Cargo.toml
sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" Cargo.toml
rm -f Cargo.toml.bak

# Verify all crate Cargo.toml files inherit or match the workspace version
ERRORS=0
for toml in crates/*/Cargo.toml apps/*/Cargo.toml; do
    [[ -f "$toml" ]] || continue
    # Crates using workspace inheritance have: version.workspace = true
    if grep -q 'version\.workspace\s*=\s*true' "$toml" 2>/dev/null; then
        continue
    fi
    # Crates with explicit version — update them
    if grep -q '^version = ' "$toml"; then
        CURRENT=$(grep '^version = ' "$toml" | head -1 | sed 's/version = "\(.*\)"/\1/')
        if [[ "$CURRENT" != "$VERSION" ]]; then
            sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" "$toml"
            rm -f "${toml}.bak"
            echo "  Updated $toml ($CURRENT → $VERSION)"
        fi
    fi
done

# Final verification: cargo metadata must show uniform version
MISMATCHED=$(cargo metadata --no-deps --format-version 1 2>/dev/null \
    | python3 -c "
import json, sys
meta = json.load(sys.stdin)
bad = [p['name'] + '=' + p['version'] for p in meta['packages']
       if p['version'] != '$VERSION' and p['name'].startswith('yscv')]
print('\n'.join(bad))
" 2>/dev/null || true)

if [[ -n "$MISMATCHED" ]]; then
    echo "" >&2
    echo "Error: The following crates do not match version $VERSION:" >&2
    echo "$MISMATCHED" >&2
    ERRORS=1
fi

if [[ "$ERRORS" -ne 0 ]]; then
    echo "" >&2
    echo "Version bump completed with errors. Fix the above and retry." >&2
    exit 1
fi

echo ""
echo "Updated workspace Cargo.toml to version $VERSION"
echo "Run 'cargo check --workspace' to verify."
