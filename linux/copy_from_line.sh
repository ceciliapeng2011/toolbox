#!/usr/bin/env sh
# Copy content of SOURCE starting at line N (1-based) into DEST.
# Usage: copy_from_line.sh N SOURCE DEST
# Example: copy_from_line.sh 15 /path/input.txt /path/output.txt

set -eu

usage() {
    echo "Usage: $0 LINE_NUMBER SOURCE_FILE DEST_FILE" >&2
    echo "  LINE_NUMBER: 1-based positive integer (1 means copy whole file)" >&2
    exit 2
}

# --- validate args ---
[ $# -eq 3 ] || usage

N="$1"
SRC="$2"
DST="$3"

# Validate N is a positive integer
case "$N" in
    ''|*[!0-9]*) echo "Error: LINE_NUMBER must be a positive integer." >&2; exit 2 ;;
    0)            echo "Error: LINE_NUMBER must be >= 1." >&2; exit 2 ;;
esac

# Validate source file
if [ ! -f "$SRC" ]; then
    echo "Error: Source file not found: $SRC" >&2
    exit 1
fi
if [ ! -r "$SRC" ]; then
    echo "Error: Source file is not readable: $SRC" >&2
    exit 1
fi

# If DEST equals SOURCE, write via a temp file to avoid truncation issues
# (sed reading and writing same file is unsafe without -i and is non-portable)
TMP=''
if [ "$(realpath "$SRC" 2>/dev/null || echo "$SRC")" = "$(realpath "$DST" 2>/dev/null || echo "$DST")" ]; then
    TMP="$(mktemp "${TMPDIR:-/tmp}/copy_from_line.XXXXXX")"
    # Use sed to print from N to end. POSIX sed supports this form.
    sed -n "${N},\$p" "$SRC" > "$TMP"
    # Move temp to original
    mv "$TMP" "$DST"
else
    # Normal case: write directly to destination (overwrites)
    sed -n "${N},\$p" "$SRC" > "$DST"
fi

# Optional: set exit status based on whether we actually wrote anything
# If N is beyond EOF, sed prints nothing; DEST will be empty (created/truncated).
# You can warn the user, but we’ll just succeed silently.
exit 0