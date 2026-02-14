# Publishing DataSpark to PyPI

## One-Time Setup

### 1. Create accounts
- **PyPI**: Register at [pypi.org](https://pypi.org/account/register/)
- **TestPyPI** (optional): Register at [test.pypi.org](https://test.pypi.org/account/register/)

### 2. Install build tools
```bash
pip install build twine
```

### 3. Configure authentication
Create `~/.pypirc`:
```ini
[pypi]
username = __token__
password = pypi-YOUR-API-TOKEN-HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR-TEST-TOKEN-HERE
```

Or use environment variables (better for CI):
```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-YOUR-API-TOKEN-HERE
```

## Before Publishing

### 1. Update version
Edit `pyproject.toml` and `dataspark/__init__.py`:
```
version = "0.1.0"  →  version = "0.2.0"
```

### 2. Update `pyproject.toml`
Replace all instances of:
- `"Your Name"` → your real name
- `"you@example.com"` → your email
- `yourusername` → your GitHub username

### 3. Run tests
```bash
pip install -e ".[dev]"
pytest
```

### 4. Check the package builds cleanly
```bash
python -m build
twine check dist/*
```

## Publish

### Test run (recommended first time)
```bash
twine upload --repository testpypi dist/*

# Verify it installs
pip install --index-url https://test.pypi.org/simple/ dataspark-ai
```

### Production release
```bash
twine upload dist/*
```

Your package is now live at: `https://pypi.org/project/dataspark-ai/`

Users install with:
```bash
pip install dataspark-ai
```

## Subsequent Releases

```bash
# 1. Bump version in pyproject.toml and __init__.py
# 2. Clean old builds
rm -rf dist/ build/ *.egg-info
# 3. Build
python -m build
# 4. Upload
twine upload dist/*
```

## CI/CD (GitHub Actions)

Create `.github/workflows/publish.yml`:
```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    permissions:
      id-token: write  # trusted publishing
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install build
      - run: python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1
```

Then enable [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) on PyPI — no tokens needed.

## Name Availability

The package name `dataspark-ai` is what I used in `pyproject.toml`. Before publishing, verify it's available:
```bash
pip index versions dataspark-ai
```
If taken, change the `name` field in `pyproject.toml` and update install instructions.
