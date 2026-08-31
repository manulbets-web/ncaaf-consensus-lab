# NCAAF Consensus Lab v3.5.17 — Posit Connect Cloud bundle

This directory was generated from:

    /Users/mgelbach/Downloads/ncaaf-consensus

Primary file for Connect Cloud: `app.py`
Python version: 3.11 or 3.12 recommended.

The deployment is public-session safe: strategy selections and alternate-line overrides are kept in each Shiny session.
PredictionTracker refresh files are runtime cache and may reset when the app sleeps or is republished.
Use Page 2's **Download prospective snapshots** button to preserve the early-week research archive outside the cloud worker.

## Local smoke test

    cd "/Users/mgelbach/Downloads/ncaaf-connect-cloud"
    python -m pip install -r requirements.txt
    python -m shiny run --reload --launch-browser app.py

## GitHub repository

The deployment target is:

    https://github.com/manulbets-web/ncaaf-consensus-lab

From this generated directory, run:

    ./push_to_github.sh

The helper initializes Git if needed, commits the deployment bundle, points `origin` at the repository above, and pushes `main` without force-pushing.

## Connect Cloud

Publish from GitHub using repository `manulbets-web/ncaaf-consensus-lab`, branch `main`, and primary file `app.py`.
