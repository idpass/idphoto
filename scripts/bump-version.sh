#!/usr/bin/env bash
# Synchronizes version across all packages in the workspace
#
# Usage: ./scripts/bump-version.sh <version>
# Example: ./scripts/bump-version.sh 0.2.0
#          ./scripts/bump-version.sh 0.2.0-alpha

set -euo pipefail

VERSION="${1:?Usage: bump-version.sh <version>}"

# Validate semver format (with optional pre-release)
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$ ]]; then
    echo "Error: Invalid version format: $VERSION"
    echo "Expected: X.Y.Z or X.Y.Z-prerelease (e.g., 1.0.0, 1.0.0-alpha, 1.0.0-beta.1)"
    exit 1
fi

echo "Bumping all packages to version $VERSION"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$ROOT_DIR"

# 1. Update workspace version in root Cargo.toml
echo "  Updating Cargo.toml workspace version..."
sed -i.bak 's/^version = ".*"/version = "'"$VERSION"'"/' Cargo.toml
rm -f Cargo.toml.bak

# 2. Update Cargo.lock
echo "  Updating Cargo.lock..."
cargo update -w --quiet

# Verify
CARGO_VERSION=$(grep -m1 '^version = ' Cargo.toml | sed 's/.*"\(.*\)".*/\1/')

if [ "$CARGO_VERSION" != "$VERSION" ]; then
    echo "  ERROR: Cargo.toml version mismatch: $CARGO_VERSION != $VERSION"
    exit 1
fi

echo ""
echo "All versions updated to $VERSION"
echo "  Cargo.toml: $CARGO_VERSION"
echo "  pyproject.toml: dynamic (reads from Cargo.toml via maturin)"
