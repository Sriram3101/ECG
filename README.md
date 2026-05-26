# Data-Poisoning-in-Healthcare

## 1) Project overview
This repository is used for experiments on data poisoning in healthcare datasets, primarily through a Google Colab notebook.

## Getting Started (quick steps)
1. Clone this repository
2. Check out the `dev` branch
3. Open the notebook in Google Colab
4. Create a feature branch before making changes

## 2) Repository structure
- `main` branch
  - Contains only `README.md` and `.gitignore`
  - Stable branch for reviewed, finalized content
  - Do not merge direct changes without review
- `dev` branch
  - Contains the active Google Colab notebook
  - All development and experimentation happen here

## 3) Branch workflow
1. Create new branches **from `dev`**
2. Make changes in your feature branch
3. Open a PR and merge back into **`dev`**
4. Merge into **`main`** only after review and stabilization

## 4) Open and use the notebook in Google Colab
1. Go to the repository on GitHub and switch to the `dev` branch
2. Open the notebook file
3. Click **Open in Colab** (or copy the GitHub URL into Colab)
4. Run cells in order and save updates

## 5) GitHub + Colab workflow
Use Colab for notebook edits and GitHub for branch/PR workflow:
- Pull the latest changes from `dev`
- Create/update your feature branch
- Commit and push from Colab (or use **File → Save a copy in GitHub** for notebook commits)
- Open PR to `dev`

## 6) Colab Git commands (clone, commit, push)
Run these in a Colab cell (prefix with `!`):

```bash
!git clone https://github.com/Nihar654/Data-Poisoning-in-Healthcare.git
%cd Data-Poisoning-in-Healthcare
!git checkout dev
!git checkout -b feature/<short-name>

# after notebook changes
!git add .
!git commit -m "feat: short description of change"
!git push -u origin feature/<short-name>
```

> **Note:** GitHub authentication from Google Colab may require a Personal Access Token (PAT) instead of your GitHub password when pushing changes.

## 7) Team collaboration guidelines
- Keep active work in `dev`-based branches only
- Keep PRs small and focused
- Review before merging
- Do not push experimental work to `main`
- Avoid editing the same notebook at the same time to reduce Jupyter/Colab merge conflicts
- Resolve review comments before merge

## 8) Commit message guidelines
Use short, clear messages:
- `feat: add baseline poisoning experiment`
- `fix: correct data split in notebook`
- `docs: update Colab workflow steps`

Format:
- `<type>: <what changed>`
- Common types: `feat`, `fix`, `docs`, `refactor`, `chore`
