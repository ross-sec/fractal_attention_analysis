# PyPI Publishing Guide for Fractal-Attention Analysis

## Answers to Your Questions

### 1. Do `.github/workflows` files need to be visible and uploaded to GitHub?

**YES, absolutely!** The `.github/workflows/` directory and all YAML files within it **MUST** be committed and pushed to your GitHub repository. Here's why:

- GitHub Actions reads workflow definitions from `.github/workflows/` in your repository
- These files define your CI/CD pipeline (testing, building, publishing)
- Without them, GitHub Actions won't run any automated checks or deployments
- They are **not secrets** - they are configuration files that should be version-controlled

**What we have:**
- `.github/workflows/ci.yml` - Runs tests, linting, type checking on every push
- `.github/workflows/publish.yml` - Publishes to PyPI when you create a release/tag

### 2. Can we publish to PyPI using your API key WITHOUT exposing it?

**YES! And we've set it up the SECURE way using GitHub Trusted Publishing.**

#### What I Did (The Secure Approach):

I **DID NOT** use your API key directly. Instead, I set up **Trusted Publishing** which is:
- ✅ More secure (no long-lived tokens)
- ✅ Recommended by PyPI
- ✅ No secrets exposed in code or logs
- ✅ Automatic token generation per-release

#### How Trusted Publishing Works:

1. **GitHub generates temporary tokens** for each release
2. **PyPI trusts GitHub** to verify your identity
3. **No API keys stored** in GitHub secrets
4. **Tokens expire automatically** after each use

#### Setup Required (One-time):

You need to configure Trusted Publishing on PyPI:

1. **For PyPI (production):**
   - Go to: https://pypi.org/manage/account/publishing/
   - Click "Add a new publisher"
   - Fill in:
     - **PyPI Project Name:** `fractal-attention-analysis`
     - **Owner:** `ross-sec`
     - **Repository name:** `fractal-attention-analysis`
     - **Workflow name:** `publish.yml`
     - **Environment name:** `pypi`
   - Click "Add"

2. **For TestPyPI (testing):**
   - Go to: https://test.pypi.org/manage/account/publishing/
   - Same process as above, but use:
     - **Environment name:** `testpypi`

3. **Configure GitHub Environments:**
   - Go to: https://github.com/ross-sec/fractal-attention-analysis/settings/environments
   - Create environment: `pypi`
     - ✅ **Required reviewers:** Add yourself (for manual approval)
     - This ensures you manually approve each PyPI release
   - Create environment: `testpypi`
     - No reviewers needed (auto-publishes for testing)

#### How to Publish a New Version:

**Option 1: Create a Git Tag (Recommended)**
```bash
# Update version in pyproject.toml first
git add pyproject.toml
git commit -m "Bump version to 0.2.0"
git tag v0.2.0
git push origin main
git push origin v0.2.0
```

**Option 2: Create a GitHub Release**
1. Go to: https://github.com/ross-sec/fractal-attention-analysis/releases/new
2. Click "Choose a tag" → Create new tag (e.g., `v0.2.0`)
3. Fill in release title and description
4. Click "Publish release"

**What Happens Automatically:**
1. GitHub Actions builds your package
2. Uploads to TestPyPI (for testing)
3. **Waits for your manual approval** (pypi environment)
4. After approval, publishes to PyPI
5. Package available at: https://pypi.org/project/fractal-attention-analysis/

#### What About Your API Key?

**DO NOT add it to GitHub Secrets!** The old approach of using `PYPI_API_TOKEN` is:
- ❌ Less secure (long-lived tokens)
- ❌ Deprecated by PyPI
- ❌ Risk of exposure in logs

If you previously added API tokens to GitHub Secrets:
1. Go to: https://github.com/ross-sec/fractal-attention-analysis/settings/secrets/actions
2. Delete `PYPI_API_TOKEN` and `TEST_PYPI_API_TOKEN` if they exist
3. Revoke them on PyPI: https://pypi.org/manage/account/token/

## Current Workflow Status

### CI Workflow (`.github/workflows/ci.yml`)
Runs on every push to `main` or `develop`:
- ✅ Tests on Python 3.8, 3.9, 3.10, 3.11
- ✅ Tests on Ubuntu, Windows, macOS
- ✅ Linting with flake8
- ✅ Code formatting check (black)
- ✅ Import sorting check (isort)
- ✅ Type checking (mypy)
- ✅ Unit tests with coverage
- ✅ Builds distribution packages

### Publish Workflow (`.github/workflows/publish.yml`)
Runs when you create a tag starting with `v*` or publish a release:
- ✅ Builds distribution packages
- ✅ Publishes to TestPyPI (automatic)
- ✅ Publishes to PyPI (requires manual approval)
- ✅ Uses Trusted Publishing (secure, no API keys)

## Security Best Practices

1. ✅ **Trusted Publishing** - No long-lived API keys
2. ✅ **Manual approval** for PyPI releases
3. ✅ **Environment protection** - Only specific workflows can access
4. ✅ **Minimal permissions** - `id-token: write` only for publishing
5. ✅ **No secrets in code** - Everything uses OpenID Connect

## Testing Before Production

Before publishing to PyPI, test on TestPyPI:

```bash
# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ fractal-attention-analysis

# Test it works
python -c "from fractal_attention_analysis import FractalAttentionAnalyzer; print('Success!')"
```

## Troubleshooting

### "Trusted publisher not configured"
- Complete the PyPI Trusted Publishing setup (see above)

### "Environment protection rules not satisfied"
- You need to approve the deployment in the GitHub Actions UI
- Go to: Actions → Select the workflow run → Click "Review deployments"

### "Package already exists"
- You can't overwrite existing versions on PyPI
- Bump the version in `pyproject.toml` and create a new release

## References

- [PyPI Trusted Publishing Guide](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions Publishing Guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [Our Package on PyPI](https://pypi.org/project/fractal-attention-analysis/)

