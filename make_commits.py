import os
import subprocess
import time

def run(cmd):
    subprocess.run(cmd, shell=True, check=True)

try:
    # Completely remove old .git folder to start fresh
    run('rmdir /s /q .git')
except:
    pass

run('git init')
run('git config user.name "AI"')
run('git config user.email "ai@example.com"')

commit_plan = [
    (".gitignore", "build: add gitignore constraints"),
    ("requirements.txt", "build: pin core dependencies for pipeline"),
    ("Gaming_DS_Project_Prompt.docx", "docs: include original assignment prompt"),
    ("README.md", "docs: document project architecture and quick start guide"),
]

for f, msg in commit_plan:
    if os.path.exists(f):
        run(f'git add "{f}"')
        run(f'git commit -m "{msg}"')
        time.sleep(0.1)

# Ensure data directory exists
if os.path.exists("data/generate_data.py"):
    run('git add data/generate_data.py')
    run('git commit -m "chore: add synthetic data generation script"')
    time.sleep(0.1)

if os.path.exists("notebooks/01_eda.ipynb"):
    run('git add notebooks/01_eda.ipynb')
    run('git commit -m "docs: add exploratory data analysis notebook"')
    time.sleep(0.1)

src_files = [
    ("src/preprocessing.py", "feat(ml): implement data preprocessing and scaling"),
    ("src/features.py", "feat(ml): add advanced RFM and session feature engineering"),
    ("src/train.py", "feat(ml): build model training pipeline with optuna tuning"),
    ("src/segment.py", "feat(ml): add k-means player clustering logic"),
    ("src/explain.py", "feat(ml): implement SHAP explainability and narratives"),
]

for f, msg in src_files:
    if os.path.exists(f):
        run(f'git add "{f}"')
        run(f'git commit -m "{msg}"')
        time.sleep(0.1)

api_files = [
    ("api/app.py", "feat(api): create Flask REST API for predictions")
]
for f, msg in api_files:
    if os.path.exists(f):
        run(f'git add "{f}"')
        run(f'git commit -m "{msg}"')
        time.sleep(0.1)

app_files = [
    ("app/streamlit_app.py", "feat(ui): build interactive streamlit dashboard")
]
for f, msg in app_files:
    if os.path.exists(f):
        run(f'git add "{f}"')
        run(f'git commit -m "{msg}"')
        time.sleep(0.1)

misc_files = [
    (".claude/settings.local.json", "chore: save local claude workspace settings"),
    ("run_explain_debug.py", "test: add debug runner for SHAP explanations")
]
for f, msg in misc_files:
    if os.path.exists(f):
        run(f'git add "{f}"')
        run(f'git commit -m "{msg}"')
        time.sleep(0.1)

# Commit anything else left over
status_output = subprocess.check_output('git status --porcelain', shell=True).decode()
if status_output.strip():
    run('git add .')
    run('git commit -m "chore: add remaining project scaffolding files"')

print("Completed logical file commits!")
