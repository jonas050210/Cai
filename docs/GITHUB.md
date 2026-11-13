# GitHub Setup

## Create a private repository

```bash
cd codeplan-lm
git init
git add .
git commit -m "Initial from-scratch CodePlan-LM pipeline"
git branch -M main
git remote add origin git@github.com:YOUR_USER/codeplan-lm.git
git push -u origin main
```

Or create a private repo in the GitHub UI and upload the files.

## Keep it small

Do not commit:
- `data/raw/`
- `data/filtered/`
- `data/packed/`
- `artifacts/`
- `checkpoints/`
- `.venv/`

These are ignored by `.gitignore`.

## GitHub Actions

The CI workflow runs lint and unit tests on CPU. It does not train the real model.
