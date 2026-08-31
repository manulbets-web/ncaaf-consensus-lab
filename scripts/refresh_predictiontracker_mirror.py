#!/usr/bin/env python3
"""Refresh PredictionTracker from a normal local internet connection and stage a GitHub mirror.

This exists because PredictionTracker currently returns HTTP 403 to Posit Connect
Cloud worker IPs. Run this on the user's Mac, where the source is reachable.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd


def iso_now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00','Z')

def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')

def sha256_bytes(b: bytes):
    return hashlib.sha256(b).hexdigest()

def run(cmd, cwd):
    p=subprocess.run(cmd,cwd=cwd,text=True,capture_output=True)
    if p.stdout: print(p.stdout, end='')
    if p.stderr: print(p.stderr, file=sys.stderr, end='')
    if p.returncode:
        raise SystemExit(p.returncode)
    return p

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('.'))
    ap.add_argument('--season', type=int, required=True)
    ap.add_argument('--week', type=int, required=True)
    args=ap.parse_args()
    root=args.root.expanduser().resolve()
    scraper=root/'scripts/scrape_predictiontracker.py'
    run([sys.executable,str(scraper),'--root',str(root),'--skip-results','--skip-archives','--strict'],root)

    manifest_path=root/'data/derived/predictiontracker_source_status.json'
    manifest=json.loads(manifest_path.read_text())
    recs=[r for r in manifest.get('records',[]) if r.get('name')=='current_predictions']
    if not recs or recs[-1].get('status')!='ok':
        raise SystemExit('Local PredictionTracker refresh did not return an OK current_predictions record.')
    rec=recs[-1]

    csv_path=root/'data/current/ncaapredictions.csv'
    frame=pd.read_csv(csv_path,low_memory=False)
    frame.columns=[str(c).strip().lower() for c in frame.columns]
    if not {'home','road','line'}.issubset(frame.columns):
        raise SystemExit('Refreshed CSV is missing home/road/line.')
    canonical=frame.to_csv(index=False,na_rep='').encode('utf-8')
    csv_path.write_bytes(canonical)
    canonical_sha=sha256_bytes(canonical)

    # Persist each unique source state in Git, not only in the ephemeral cloud worker.
    snap_dir=root/'data/snapshots/predictiontracker/mirror'/f'season_{args.season}'/f'week_{args.week:02d}'
    snap_dir.mkdir(parents=True,exist_ok=True)
    snap_path=snap_dir/f'{stamp()}_{canonical_sha[:12]}_ncaapredictions.csv'
    existing=list(snap_dir.glob(f'*_{canonical_sha[:12]}_ncaapredictions.csv'))
    if not existing:
        snap_path.write_bytes(canonical)
    else:
        snap_path=existing[-1]

    page_validation={}
    try:
        msg=json.loads(rec.get('message') or '{}')
        page_validation=msg.get('page_validation') or {}
    except Exception:
        pass

    meta={
        'mirror_schema': 1,
        'season': args.season,
        'week': args.week,
        'fetched_at_utc': rec.get('fetched_at_utc') or iso_now(),
        'published_update': rec.get('published_update'),
        'canonical_sha256': canonical_sha,
        'rows': int(len(frame)),
        'columns': int(len(frame.columns)),
        'snapshot_path': str(snap_path.relative_to(root)),
        'page_validation': page_validation,
        'source_url': rec.get('url'),
        'note': 'Fetched locally because PredictionTracker blocks Posit Connect Cloud worker IPs.',
    }
    meta_path=root/'data/current/predictiontracker_mirror_status.json'
    meta_path.write_text(json.dumps(meta,indent=2)+'\n')

    print('\nPredictionTracker mirror staged successfully')
    print(f'  season/week: {args.season}/{args.week}')
    print(f'  games:       {len(frame)}')
    print(f'  source hash: {canonical_sha[:12]}')
    print(f'  published:   {rec.get("published_update") or "unknown"}')
    print(f'  snapshot:    {snap_path.relative_to(root)}')
    print(f'  metadata:    {meta_path.relative_to(root)}')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
