#!/bin/sh

# Format all staged Python files (that are NOT deleted)
PYTHON_FILES=$(git diff --cached --name-only --diff-filter=d | grep -E '\.py')
if [ -n "$PYTHON_FILES" ]; then
    for file in $PYTHON_FILES; do
        black "$file" && git add "$file"
    done
fi
