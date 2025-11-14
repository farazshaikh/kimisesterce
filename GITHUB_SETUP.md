# GitHub Repository Setup Guide

## Quick Setup

Follow these steps to create the `kimisesterce` repository on GitHub:

### Step 1: Authenticate with GitHub

```bash
gh auth login
```

**Follow the prompts:**
1. Choose: **GitHub.com**
2. Choose: **HTTPS** (recommended) or SSH
3. Authentication method:
   - **Option A (Easiest)**: Login with a web browser
   - **Option B**: Paste an authentication token

**For Option A (Web Browser):**
- gh will show you a code
- Open the URL in your browser
- Enter the code
- Authorize the app

**For Option B (Token):**
- Go to: https://github.com/settings/tokens
- Click "Generate new token" → "Generate new token (classic)"
- Select scopes: `repo`, `read:org`, `workflow`
- Copy the token
- Paste it when gh asks

### Step 2: Create and Push Repository

Once authenticated, run:

```bash
./setup_github.sh
```

This will:
- ✓ Create the repository `farazshaikh/kimisesterce` on GitHub
- ✓ Set it as the origin remote
- ✓ Push your code to GitHub
- ✓ Make it publicly accessible

---

## Manual Setup (Alternative)

If you prefer to do it manually:

```bash
# 1. Authenticate
gh auth login

# 2. Create the repository
gh repo create farazshaikh/kimisesterce \
  --public \
  --source=. \
  --remote=origin \
  --description="Kimi K2 Instruct setup for 8x B200 GPUs with vLLM" \
  --push
```

---

## Verify Setup

After setup, verify with:

```bash
# Check remote
git remote -v

# View on GitHub
gh repo view --web
```

---

## Repository Details

- **Owner**: farazshaikh
- **Name**: kimisesterce  
- **URL**: https://github.com/farazshaikh/kimisesterce
- **Visibility**: Public
- **Description**: Kimi K2 Instruct setup for 8x B200 GPUs with vLLM

---

## What Gets Pushed

The following files will be pushed to GitHub:

✅ `init_baremetal.sh` - Main setup script
✅ `README.md` - Quick start guide
✅ `B200_TROUBLESHOOTING.md` - B200 GPU troubleshooting
✅ `.gitignore` - Git ignore rules
✅ `setup_github.sh` - This GitHub setup script
✅ `GITHUB_SETUP.md` - This guide

❌ `logs/` - Excluded (in .gitignore)
❌ `.venv/` - Excluded (in .gitignore)
❌ `__pycache__/` - Excluded (in .gitignore)

---

## Troubleshooting

### "gh: command not found"
```bash
sudo apt install gh
```

### "You are not logged into any GitHub hosts"
```bash
gh auth login
```

### "Permission denied (publickey)"
```bash
# Switch to HTTPS authentication
gh auth login

# Or set up SSH keys:
ssh-keygen -t ed25519 -C "your_email@example.com"
gh ssh-key add ~/.ssh/id_ed25519.pub
```

### "Repository already exists"
If the repo already exists, the script will ask if you want to add it as a remote and push.

---

## Next Steps

After pushing to GitHub:

1. **Add collaborators** (if needed):
   ```bash
   gh repo edit --add-collaborator username
   ```

2. **Enable GitHub Actions** (optional):
   - Add `.github/workflows/` directory
   - Create CI/CD workflows

3. **Add topics** (optional):
   ```bash
   gh repo edit --add-topic vllm --add-topic kimi --add-topic llm --add-topic gpu
   ```

4. **Update repository settings**:
   ```bash
   gh repo edit --description "Your new description"
   gh repo edit --homepage "https://your-site.com"
   ```

---

## Summary

**Quick commands:**
```bash
# 1. Authenticate (one-time)
gh auth login

# 2. Create and push repo
./setup_github.sh

# 3. View on GitHub
gh repo view --web
```

That's it! Your Kimi K2 setup will be available on GitHub! 🚀

