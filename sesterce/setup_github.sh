#!/bin/bash

################################################################################
# GitHub Repository Setup Script
# Creates and pushes the kimisesterce repository to GitHub
################################################################################

set -e

GITHUB_USER="farazshaikh"
REPO_NAME="kimisesterce"
REPO_DESCRIPTION="Kimi K2 Instruct setup for 8x B200 GPUs with vLLM"

echo "=========================================="
echo "GitHub Repository Setup"
echo "=========================================="
echo ""
echo "Repository: $GITHUB_USER/$REPO_NAME"
echo "Description: $REPO_DESCRIPTION"
echo ""

# Check if gh is installed
if ! command -v gh &> /dev/null; then
    echo "Error: GitHub CLI (gh) is not installed."
    echo "Install it with: sudo apt install gh"
    exit 1
fi

# Check authentication
echo "Checking GitHub authentication..."
if ! gh auth status &> /dev/null; then
    echo ""
    echo "❌ You are not authenticated with GitHub."
    echo ""
    echo "Please authenticate by running:"
    echo "  gh auth login"
    echo ""
    echo "After authentication, run this script again."
    exit 1
fi

echo "✓ GitHub authentication verified"
echo ""

# Check if repo already exists on GitHub
echo "Checking if repository already exists..."
if gh repo view "$GITHUB_USER/$REPO_NAME" &> /dev/null; then
    echo ""
    echo "⚠️  Repository $GITHUB_USER/$REPO_NAME already exists on GitHub."
    echo ""
    read -p "Do you want to add it as a remote and push? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
    
    # Add remote if it doesn't exist
    if ! git remote get-url origin &> /dev/null; then
        echo "Adding remote origin..."
        gh repo set-default "$GITHUB_USER/$REPO_NAME"
        git remote add origin "git@github.com:$GITHUB_USER/$REPO_NAME.git"
    fi
else
    # Create new repository
    echo "Creating new GitHub repository..."
    gh repo create "$GITHUB_USER/$REPO_NAME" \
        --public \
        --source=. \
        --remote=origin \
        --description="$REPO_DESCRIPTION" \
        --push
    
    echo ""
    echo "✓ Repository created successfully!"
    echo ""
    echo "🔗 Repository URL: https://github.com/$GITHUB_USER/$REPO_NAME"
    exit 0
fi

# Push to existing repo
echo ""
echo "Pushing to GitHub..."
git push -u origin master

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "🔗 Repository URL: https://github.com/$GITHUB_USER/$REPO_NAME"
echo ""
echo "View your repo:"
echo "  gh repo view --web"
echo ""

