#!/bin/bash

# Script to count total number of lines written across all commits in git repository history
# Usage: ./count_total_lines.sh [repository_path]

set -e  # Exit on any error

# Default to current directory if no path provided
REPO_PATH="${1:-.}"

# Check if we're in a git repository
if [ ! -d "$REPO_PATH/.git" ]; then
    echo "Error: $REPO_PATH is not a git repository"
    exit 1
fi

cd "$REPO_PATH"

echo "Counting total lines written across all commits in repository: $(pwd)"
echo "This may take a while for repositories with long history..."
echo

# Initialize counters
total_added=0
total_removed=0
commit_count=0

# Get all commits in reverse chronological order and process them
git rev-list --all --reverse | while read commit; do
    # Get the stats for this commit (lines added and removed)
    stats=$(git show --numstat --format="" "$commit" 2>/dev/null | awk '{added+=$1; removed+=$2} END {print added+0, removed+0}')
    
    if [ ! -z "$stats" ]; then
        added=$(echo $stats | cut -d' ' -f1)
        removed=$(echo $stats | cut -d' ' -f2)
        
        # Skip if stats are invalid (binary files show as -)
        if [[ "$added" =~ ^[0-9]+$ ]] && [[ "$removed" =~ ^[0-9]+$ ]]; then
            total_added=$((total_added + added))
            total_removed=$((total_removed + removed))
        fi
    fi
    
    commit_count=$((commit_count + 1))
    
    # Show progress every 100 commits
    if [ $((commit_count % 100)) -eq 0 ]; then
        echo "Processed $commit_count commits..."
    fi
done > /tmp/git_line_count.tmp

# Read the final results
total_added=0
total_removed=0
commit_count=0

# Process commits again to get final totals (since the while loop runs in a subshell)
git rev-list --all | while read commit; do
    stats=$(git show --numstat --format="" "$commit" 2>/dev/null | awk '{added+=$1; removed+=$2} END {print added+0, removed+0}')
    
    if [ ! -z "$stats" ]; then
        added=$(echo $stats | cut -d' ' -f1)
        removed=$(echo $stats | cut -d' ' -f2)
        
        if [[ "$added" =~ ^[0-9]+$ ]] && [[ "$removed" =~ ^[0-9]+$ ]]; then
            echo "$added $removed"
        fi
    fi
done | awk '{total_added+=$1; total_removed+=$2} END {
    printf "=== RESULTS ===\n"
    printf "Total commits processed: %d\n", NR
    printf "Total lines added: %s\n", total_added
    printf "Total lines removed: %s\n", total_removed
    printf "Net lines written: %s\n", total_added
    printf "Total line changes: %s\n", total_added + total_removed
}'

echo
echo "Note: This counts all line additions across history, including:"
echo "- Lines that were later modified or deleted"
echo "- Lines in binary files are excluded"
echo "- Merge commits are included"