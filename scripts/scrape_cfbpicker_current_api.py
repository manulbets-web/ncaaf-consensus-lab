#!/usr/bin/env python3
"""
Current-week CFB Picker collector.

Uses the proven Tableau Embedding API v3 + L# canvas/tooltip extraction used
by the historical collector. Tableau is filtered only by year and picker by
default; exact current-week membership is enforced against PredictionTracker's
current master slate after tooltip extraction.
"""
from __future__ import annotations

import argparse
import contextlib
import difflib
import http.server
import importlib.util
import json
import re
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import numpy as np
import pandas as pd

TABLEAU_BASE = "https://public.tableau.com/views/CFBPicker"
TABLEAU_API = "https://public.tableau.com/javascripts/api/tableau.embedding.3.latest.min.js"
COLLECTOR_VERSION = "v3.5.38-tableau-embedding-api-current"


def iso_utc() -> str:
    return pd.Timestamp.utcnow().isoformat()


def safe_key(value: str) -> str:
    text = str(value).lower().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "unknown"


def load_history_module(root: Path) -> Any:
    path = Path(__file__).with_name("cfbpicker_tooltip_legacy.py")
    if not path.exists():
        raise FileNotFoundError(
            "The proven CFB Picker tooltip helper is required but is missing: "
            f"{path}"
        )
    spec = importlib.util.spec_from_file_location("cfbpicker_history_runtime_current", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass


@contextlib.contextmanager
def local_server(directory: Path):
    port = free_port()
    def handler(*args: Any, **kwargs: Any) -> QuietHandler:
        return QuietHandler(*args, directory=str(directory), **kwargs)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/index.html"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def embedding_html() -> str:
    template = r'''<!doctype html>
<html><head><meta charset="utf-8"><title>CFB Picker current API collector</title>
<style>html,body{margin:0;width:100%;height:100%;overflow:auto}#host,tableau-viz{width:1700px;height:1250px}</style>
</head><body><div id="host"></div>
<script type="module">
import { TableauViz, TableauEventType, FilterUpdateType } from "__TABLEAU_API__";
const query = new URLSearchParams(window.location.search);
const requestedYear = Number(query.get("year"));
const requestedWeek = query.get("week");
const requestedPicker = query.get("picker");
const requestedComplete = query.get("complete");
window.__cfbp = {status:"initializing",requestedYear,requestedWeek,requestedPicker,requestedComplete,attempts:[],errors:[]};
function serialize(value){
  if(value===null||value===undefined)return value;
  if(["string","number","boolean"].includes(typeof value))return value;
  if(Array.isArray(value))return value.map(serialize);
  const output={};
  try{for(const key of Object.keys(value)){try{output[key]=serialize(value[key]);}catch(error){output[key]=`[unreadable: ${error}]`;}}return output;}catch(error){return String(value);}
}
function targets(activeSheet){const output=[activeSheet];if(Array.isArray(activeSheet.worksheets))output.push(...activeSheet.worksheets);return output;}
async function applyCategorical(activeSheet,fieldName,value){
  const errors=[];
  for(const target of targets(activeSheet)){
    try{const result=await target.applyFilterAsync(fieldName,[String(value)],FilterUpdateType.Replace);window.__cfbp.attempts.push({target:target.name,fieldName,value:String(value),result:serialize(result)});return true;}
    catch(error){errors.push(`${target.name}: ${error}`);}
  }
  window.__cfbp.attempts.push({fieldName,value,errors});return false;
}
async function filterSummary(sheet){
  try{const filters=await sheet.getFiltersAsync();return filters.map((filter)=>({fieldName:filter.fieldName,fieldId:filter.fieldId,filterType:filter.filterType,worksheetName:filter.worksheetName,appliedValues:serialize(filter.appliedValues),isAllSelected:filter.isAllSelected}));}
  catch(error){return [{error:String(error)}];}
}
async function collectAllFilters(activeSheet){const output=[];for(const target of targets(activeSheet)){const filters=await filterSummary(target);for(const filter of filters)output.push({...filter,targetName:target.name});}return output;}
function formattedValues(filters,fieldName){const output=[];for(const filter of filters){if(filter.fieldName!==fieldName)continue;for(const value of (filter.appliedValues||[])){output.push(String(value?._formattedValue??value?._nativeValue??value?._value));}}return output;}
const source=new URL("__TABLEAU_BASE__/L");
source.searchParams.set(":showVizHome","no");
source.searchParams.set("Perspective","Forward");
source.searchParams.set("Show Incomplete","Yes");
source.searchParams.set("Show Line","Yes");
source.searchParams.set("Vs Line X","Close");
const viz=new TableauViz();viz.src=source.toString();viz.toolbar="hidden";viz.hideTabs=false;viz.width="1700px";viz.height="1250px";
viz.addEventListener(TableauEventType.VizLoadError,(event)=>{window.__cfbp.status="viz_load_error";window.__cfbp.errors.push(String(event?.detail||event));});
viz.addEventListener(TableauEventType.FirstInteractive,async()=>{
  try{
    const workbook=viz.workbook;const activeSheet=workbook.activeSheet;window.__cfbp.status="interactive";
    window.__cfbp.activeSheet={name:activeSheet.name,sheetType:activeSheet.sheetType,worksheetNames:Array.isArray(activeSheet.worksheets)?activeSheet.worksheets.map((item)=>item.name):[]};
    await new Promise((resolve)=>setTimeout(resolve,1200));
    const yearApplied=await applyCategorical(activeSheet,"Year ",requestedYear);await new Promise((resolve)=>setTimeout(resolve,700));
    const hasWeekRequest=requestedWeek!==null&&requestedWeek!=="";
    const weekApplied=hasWeekRequest?await applyCategorical(activeSheet,"Week ",requestedWeek):true;await new Promise((resolve)=>setTimeout(resolve,500));
    const pickerApplied=await applyCategorical(activeSheet,"Picker",requestedPicker);await new Promise((resolve)=>setTimeout(resolve,500));
    const hasCompleteRequest=requestedComplete!==null&&requestedComplete!=="";
    const completeApplied=hasCompleteRequest?await applyCategorical(activeSheet,"Game Complete",requestedComplete):true;await new Promise((resolve)=>setTimeout(resolve,2500));
    const filters=await collectAllFilters(activeSheet);
    const yearValues=formattedValues(filters,"Year ");const weekValues=formattedValues(filters,"Week ");const pickerValues=formattedValues(filters,"Picker");const completeValues=formattedValues(filters,"Game Complete");
    const yearVerified=yearValues.includes(String(requestedYear));const weekVerified=!hasWeekRequest||weekValues.includes(String(requestedWeek));const pickerVerified=pickerValues.includes(String(requestedPicker));const completeVerified=!hasCompleteRequest||completeValues.includes(String(requestedComplete));
    window.__cfbp.filtersAfter=filters;window.__cfbp.yearValues=yearValues;window.__cfbp.weekValues=weekValues;window.__cfbp.pickerValues=pickerValues;window.__cfbp.completeValues=completeValues;
    window.__cfbp.applied={yearApplied,weekApplied,pickerApplied,completeApplied};window.__cfbp.verified={yearVerified,weekVerified,pickerVerified,completeVerified};
    window.__cfbp.status=(yearApplied&&weekApplied&&pickerApplied&&yearVerified&&weekVerified&&pickerVerified&&completeVerified)?"ready":"filter_verification_failed";
  }catch(error){window.__cfbp.status="error";window.__cfbp.errors.push(`${error?.name||"Error"}: ${error?.message||error}`);}
});
document.getElementById("host").appendChild(viz);window.__tableauViz=viz;
</script></body></html>'''
    return template.replace("__TABLEAU_API__", TABLEAU_API).replace("__TABLEAU_BASE__", TABLEAU_BASE)


def tableau_frame(page: Any) -> Any:
    needle = "/views/CFBPicker/L".lower()
    for frame in page.frames:
        if needle in frame.url.lower():
            return frame
    raise RuntimeError("Could not locate Tableau L frame.")


def week_filter_candidates(week: int, override: str | None = None) -> list[str | None]:
    """Do not filter Tableau by week unless the caller explicitly overrides.

    The displayed ``Wk 1-2`` title is not a categorical filter value, and the
    workbook's calculated ``Week `` field does not reliably share the pipeline's
    numbering. Current-slate membership is enforced after extraction against
    PredictionTracker instead.
    """
    if override is not None and str(override).strip():
        return [str(override).strip()]
    return [None]


def open_filtered_view(
    context: Any, server_url: str, year: int, week: int, picker: str,
    wait_seconds: int, week_filter_override: str | None = None,
    completion_filter_override: str | None = None,
):
    failures = []
    for week_filter in week_filter_candidates(week, week_filter_override):
        query = {"year": str(year), "picker": picker}
        if week_filter is not None:
            query["week"] = week_filter
        if completion_filter_override is not None and str(completion_filter_override).strip():
            query["complete"] = str(completion_filter_override).strip()
        local_url = f"{server_url}?{urlencode(query)}"
        page = context.new_page(); page.set_default_timeout(20_000)
        try:
            page.goto(local_url, wait_until="domcontentloaded", timeout=90_000)
            deadline = time.monotonic() + 120; state = {}; terminal={"ready","filter_verification_failed","viz_load_error","error"}
            while time.monotonic() < deadline:
                state = page.evaluate("() => window.__cfbp || {}")
                if state.get("status") in terminal: break
                page.wait_for_timeout(250)
            if state.get("status") != "ready":
                failures.append({"week_filter": week_filter, "state": state})
                page.close()
                continue
            state["logical_week"] = int(week)
            state["resolved_week_filter"] = week_filter
            state["resolved_completion_filter"] = (
                str(completion_filter_override).strip()
                if completion_filter_override is not None
                and str(completion_filter_override).strip()
                else None
            )
            page.wait_for_timeout(max(1, wait_seconds) * 1000)
            return page, tableau_frame(page), state, local_url
        except Exception as exc:
            failures.append({
                "week_filter": week_filter,
                "error": f"{type(exc).__name__}: {exc}",
            })
            page.close()
    raise RuntimeError(
        "Current Tableau API filters were not verified: "
        + json.dumps(failures, default=str)[:8000]
    )


def normalize_team(value: str) -> str:
    x=str(value or "").lower().replace("&"," and ");x=re.sub(r"\bstate\b","st",x);x=re.sub(r"\bsaint\b","st",x);x=re.sub(r"[^a-z0-9]","",x)
    aliases={"miamifl":"miami","miamiflorida":"miami","connecticut":"uconn","massachusetts":"umass","southerncalifornia":"usc","southerncal":"usc"}
    return aliases.get(x,x)


def displayed_matches(displayed: str, full: str) -> bool:
    d_raw=str(displayed or "").strip();f_raw=str(full or "").strip();d=normalize_team(d_raw.replace("..",""));f=normalize_team(f_raw)
    if not d or not f:return False
    if d==f:return True
    return ".." in d_raw and len(d)>=4 and f.startswith(d)


def load_master_slate(root: Path) -> pd.DataFrame:
    path=root/"data/current/ncaapredictions.csv"
    if not path.exists():return pd.DataFrame()
    x=pd.read_csv(path,low_memory=False);x.columns=[str(c).strip().lower() for c in x.columns]
    if not {"road","home"}.issubset(x.columns):return pd.DataFrame()
    y=x[[c for c in ["road","home","line"] if c in x.columns]].copy();y["road"]=y["road"].astype(str);y["home"]=y["home"].astype(str)
    y["market_home_margin"]=pd.to_numeric(y["line"],errors="coerce") if "line" in y.columns else np.nan
    return y[["road","home","market_home_margin"]]


def resolve_header_game(header_values: list[str], master: pd.DataFrame):
    if len(header_values)<2:raise ValueError(f"Expected Away/Home header cells, got {header_values}")
    a=str(header_values[0]).strip();h=str(header_values[1]).strip()
    if master.empty:return a,h,None,None
    matches=master[master.apply(lambda r: displayed_matches(a,r["road"]) and displayed_matches(h,r["home"]),axis=1)]
    if len(matches)==1:
        r=matches.iloc[0];m=pd.to_numeric(pd.Series([r["market_home_margin"]]),errors="coerce").iloc[0]
        return str(r["road"]),str(r["home"]),float(m) if np.isfinite(m) else None,True
    return a,h,None,False


def label_team_matches(label: str|None, team: str) -> bool:
    if label is None:return False
    a=normalize_team(label);b=normalize_team(team)
    if a==b:return True
    return ".." in str(label) and len(a)>=4 and b.startswith(a)


def _team_similarity(label: str | None, candidate: str) -> float:
    """Score a tooltip team label against one game-side candidate.

    Tableau frequently abbreviates directionals (``E. Michigan``) and team
    names independently of the row header. This score is used only to choose
    between the already-resolved away and home teams for one game.
    """
    if label is None:
        return 0.0
    a = normalize_team(label)
    b = normalize_team(candidate)
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(a) >= 4 and (a.startswith(b) or b.startswith(a)):
        return 0.96
    directional = {
        "e": "eastern", "w": "western", "n": "northern", "s": "southern",
    }
    for short, long in directional.items():
        if a.startswith(short) and b.startswith(long):
            a_tail = a[len(short):]
            b_tail = b[len(long):]
            if a_tail and b_tail and (
                a_tail == b_tail
                or (min(len(a_tail), len(b_tail)) >= 4
                    and (a_tail.startswith(b_tail) or b_tail.startswith(a_tail)))
            ):
                return 0.92
        if b.startswith(short) and a.startswith(long):
            b_tail = b[len(short):]
            a_tail = a[len(long):]
            if a_tail and b_tail and (
                a_tail == b_tail
                or (min(len(a_tail), len(b_tail)) >= 4
                    and (a_tail.startswith(b_tail) or b_tail.startswith(a_tail)))
            ):
                return 0.92
    return float(difflib.SequenceMatcher(None, a, b).ratio())


def resolve_labeled_team_side(
    label: str | None,
    away: str,
    home: str,
    header_values: list[str] | None = None,
) -> str | None:
    if label_team_matches(label, home):
        return "home"
    if label_team_matches(label, away):
        return "away"
    away_candidates = [away]
    home_candidates = [home]
    if header_values and len(header_values) >= 2:
        away_candidates.append(str(header_values[0]).replace("..", ""))
        home_candidates.append(str(header_values[1]).replace("..", ""))
    away_score = max(_team_similarity(label, value) for value in away_candidates)
    home_score = max(_team_similarity(label, value) for value in home_candidates)
    best = max(away_score, home_score)
    gap = abs(home_score - away_score)
    if best < 0.58 or gap < 0.12:
        return None
    return "home" if home_score > away_score else "away"


def favorite_to_home_margin_current(
    team: str | None,
    value: float | None,
    away: str,
    home: str,
    header_values: list[str] | None = None,
):
    if value is None:return None
    value=float(value)
    if abs(value)<1e-12:return 0.0
    side = resolve_labeled_team_side(team, away, home, header_values)
    if side == "home":return abs(value)
    if side == "away":return -abs(value)
    raise ValueError(f"Favorite {team!r} did not match {away!r} / {home!r}.")


def parse_current_tooltip(text: str, header_values: list[str], master: pd.DataFrame, history: Any):
    picker_match=re.search(r"^\s*Picker:\s*(.*?)\s*$",text,flags=re.MULTILINE)
    person_match=re.search(r"^\s*Person:\s*(.*?)\s*$",text,flags=re.MULTILINE)
    twitter_match=re.search(r"^\s*Twitter:\s*(.*?)\s*$",text,flags=re.MULTILINE)
    line_team,line_value=history.labeled_team_margin(text,"Line");pick_team,pick_value=history.labeled_team_margin(text,"Pick")
    if picker_match is None or pick_team is None or pick_value is None:raise ValueError("Current tooltip did not expose Picker + Pick fields:\n"+text)
    away,home,pt_market,master_slate_match=resolve_header_game(header_values,master)
    prediction=favorite_to_home_margin_current(pick_team,pick_value,away,home,header_values)
    tooltip_market=favorite_to_home_margin_current(line_team,line_value,away,home,header_values) if line_value is not None else None
    market=pt_market if pt_market is not None and np.isfinite(pt_market) else tooltip_market
    return {"picker":picker_match.group(1).strip(),"person":person_match.group(1).strip() if person_match else None,"twitter":twitter_match.group(1).strip() if twitter_match else None,"away":away,"home":home,"market_home_margin_close":market,"prediction_home_margin":prediction,"pick_team":pick_team,"pick_value":pick_value,"line_team":line_team,"line_value":line_value,"master_slate_match":master_slate_match}


def scrape_picker(context,server_url,root,history,master,*,season,week,picker,canonical_model_id,model_name,wait_seconds,screenshot_errors,week_filter_override=None,completion_filter_override=None):
    page=None;record={"canonical_model_id":canonical_model_id,"model_name":model_name,"picker":picker,"season":int(season),"week":int(week),"status":"error","rows_detected":0,"rows_extracted":0,"rows_failed":0,"row_errors":[],"message":None,"filter_state":None}
    raw_dir=root/"data/raw/cfbpicker/current_api"/f"season={season}"/f"picker={safe_key(picker)}";raw_dir.mkdir(parents=True,exist_ok=True);checkpoint=raw_dir/f"week={week:02d}.csv"
    try:
        page,frame,state,local_url=open_filtered_view(context,server_url,season,week,picker,wait_seconds,week_filter_override,completion_filter_override);record["filter_state"]=state;record["url"]=local_url;record["resolved_week_filter"]=state.get("resolved_week_filter");record["resolved_completion_filter"]=state.get("resolved_completion_filter")
        try:frame.add_style_tag(content=".tab-tooltip,.tab-tooltipContent,[class*='tooltip' i]{pointer-events:none!important;}")
        except Exception:pass
        view=frame.locator(".tab-tvView").first;canvas=view.locator("canvas.tabCanvas").first;canvas_box=canvas.bounding_box();zone=canvas.locator("xpath=ancestor::div[contains(@class, 'tabZone-viz')][1]").first
        if zone.count()==0:zone=frame.locator("body")
        if not canvas_box:
            body=frame.locator("body").inner_text(timeout=5_000)
            if re.search(r"No data|No records|No matching|Games\s*:\s*None",body,re.I):pd.DataFrame().to_csv(checkpoint,index=False);record["status"]="no_data";return pd.DataFrame(),record
            raise RuntimeError("Could not locate current API-filtered L# canvas.")
        rows=history.collect_header_rows(zone,canvas_box);record["rows_detected"]=int(len(rows))
        if not rows:pd.DataFrame().to_csv(checkpoint,index=False);record["status"]="no_data";return pd.DataFrame(),record
        extracted=[];row_errors=[]
        for row_number,row in enumerate(rows,start=1):
            anchor_id=row.get("anchor_id")
            if anchor_id:
                try:frame.locator(f"#{anchor_id}").scroll_into_view_if_needed();page.wait_for_timeout(100)
                except Exception:pass
            current_y=row["center_y"]
            if anchor_id:
                try:
                    anchor_box=frame.locator(f"#{anchor_id}").bounding_box()
                    if anchor_box:current_y=anchor_box["y"]+anchor_box["height"]/2
                except Exception:pass
            current_canvas_box=canvas.bounding_box()
            if not current_canvas_box:raise RuntimeError(f"Canvas disappeared before row {row_number}.")
            parsed=None;tooltip="";notes=[];candidate_previews=[]
            for xf in (0.50,0.30,0.70,0.15,0.85):
                if parsed is not None:break
                for dy in (0.0,-2.0,2.0,-5.0,5.0):
                    try:page.keyboard.press("Escape")
                    except Exception:pass
                    page.mouse.move(max(1.0,current_canvas_box["x"]-12.0),max(1.0,current_canvas_box["y"]-8.0));page.wait_for_timeout(80)
                    x=current_canvas_box["x"]+current_canvas_box["width"]*xf
                    try:candidate=history.click_and_read_tooltip_response(page,x=x,y=current_y+dy)
                    except Exception as exc:notes.append(f"x={xf:.2f},dy={dy:+.1f}:{type(exc).__name__}");continue
                    if not candidate:notes.append(f"x={xf:.2f},dy={dy:+.1f}:empty");continue
                    try:cp=parse_current_tooltip(candidate,row["header_values"],master,history)
                    except Exception as exc:
                        notes.append(f"x={xf:.2f},dy={dy:+.1f}:parse-{type(exc).__name__}: {exc}")
                        preview=candidate.strip()
                        if preview and preview not in candidate_previews:
                            candidate_previews.append(preview[:2000])
                        continue
                    if safe_key(cp["picker"])!=safe_key(picker):notes.append(f"picker={cp['picker']}");continue
                    parsed=cp;tooltip=candidate;break
            if parsed is None:
                row_errors.append({
                    "row_number": int(row_number),
                    "header_values": [str(value) for value in row["header_values"]],
                    "attempts": notes[-12:],
                    "candidate_tooltips": candidate_previews[-3:],
                })
                continue
            extracted.append({"season":int(season),"week":int(week),"picker":picker,"canonical_model_id":canonical_model_id,"model_name":model_name,"person":parsed["person"],"twitter":parsed["twitter"],"away":parsed["away"],"home":parsed["home"],"market_home_margin_close":parsed["market_home_margin_close"],"prediction_home_margin":parsed["prediction_home_margin"],"pick_team":parsed["pick_team"],"pick_value":parsed["pick_value"],"line_team":parsed["line_team"],"line_value":parsed["line_value"],"master_slate_match":parsed["master_slate_match"],"filter_method":"tableau_embedding_api_master_slate","scraped_at_utc":iso_utc(),"row_number":row_number,"header_values":" | ".join(row["header_values"]),"tooltip_text":tooltip.replace("\n","\\n")})
        record["rows_failed"]=int(len(row_errors));record["row_errors"]=row_errors
        if not extracted and row_errors:
            record["message"]=(f"All {len(row_errors)} detected rows failed tooltip parsing; "
                               "see row_errors for exact tooltip text and exceptions.")
            return pd.DataFrame(),record
        data=pd.DataFrame(extracted).drop_duplicates(["season","week","picker","away","home"])
        record["rows_parsed_before_master_slate"]=int(len(data))
        if not master.empty:
            data=data[data["master_slate_match"].eq(True)].copy()
        data.to_csv(checkpoint,index=False)
        record["rows_master_slate"]=int(len(data))
        if data.empty:
            record["status"]="no_data";return pd.DataFrame(),record
        record["status"]="ok";record["rows_extracted"]=int(len(data));record["checkpoint"]=str(checkpoint.relative_to(root))
        if row_errors:
            record["message"]=(f"Collected {len(data)} current-slate rows; skipped "
                               f"{len(row_errors)} of {len(rows)} detected Tableau rows. "
                               "See row_errors for the audited failures.")
        return data,record
    except Exception as exc:
        record["message"]=f"{type(exc).__name__}: {exc}"
        if screenshot_errors and page is not None:
            err=root/"data/raw/cfbpicker/current_api/errors"/f"season={season}"/f"picker={safe_key(picker)}";err.mkdir(parents=True,exist_ok=True)
            try:page.screenshot(path=str(err/f"week={week:02d}.png"),full_page=True);(err/f"week={week:02d}_state.json").write_text(json.dumps(page.evaluate("() => window.__cfbp || {}"),indent=2,default=str),encoding="utf-8")
            except Exception:pass
        return pd.DataFrame(),record
    finally:
        if page is not None:page.close()


def _mapping_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    mapping = pd.read_csv(path, low_memory=False)
    if "picker_name" not in mapping.columns and "picker" in mapping.columns:
        mapping = mapping.rename(columns={"picker": "picker_name"})
    required = {"canonical_model_id", "picker_name", "model_name"}
    if not required.issubset(mapping.columns):
        return pd.DataFrame()
    keep = [c for c in [
        "canonical_model_id", "model_name", "picker_name",
        "is_new_canonical_model", "mapping_reason",
    ] if c in mapping.columns]
    mapping = mapping[keep].dropna(
        subset=["canonical_model_id", "model_name", "picker_name"]
    ).drop_duplicates()
    for col in ["canonical_model_id", "model_name", "picker_name"]:
        mapping[col] = mapping[col].astype(str).str.strip()
    return mapping


def _picker_names(path: Path) -> list[str]:
    values = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        value = raw.strip()
        if value and not value.startswith("#"):
            values.append(value)
    return list(dict.fromkeys(values))


def _compact_model(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _canonical_live_picker_name(value: str) -> str:
    groups = {
        "FEI": ["FEI"],
        "SP+": ["SP+", "SP Plus", "SPPlus", "Bill Connelly SP+"],
        "FPI": ["FPI", "ESPN FPI"],
        "KFord": ["KFord", "KFord Ratings", "Kelley Ford", "Kelley Ford Ratings"],
        "Massey Ratings": ["Massey", "Massey Ratings"],
        "Harville": ["Harville", "David Harville"],
        "TeamRankings": ["TeamRankings", "TeamRankings.com", "Team Rankings"],
        "Sagarin: Predictor": ["Sagarin Predictor", "Sagarin: Predictor", "Sagarin Pred"],
        "Sagarin: Golden": ["Sagarin Golden Mean", "Sagarin: Golden", "Sagarin Golden"],
        "Sagarin": ["Sagarin", "Sagarin Ratings"],
        "Sagarin: Recent": ["Sagarin Recent", "Sagarin: Recent"],
        "Slate Fluker": ["Slate Fluker", "Slate Index"],
    }
    key = _compact_model(value)
    for canonical, aliases in groups.items():
        if key in {_compact_model(alias) for alias in aliases}:
            return canonical
    return str(value).strip()


def _model_aliases(value: str) -> set[str]:
    return {
        key for key in {
            _compact_model(value),
            _compact_model(_canonical_live_picker_name(value)),
        } if key
    }


def _dynamic_live_mapping(root: Path, live_pickers: list[str]) -> pd.DataFrame:
    pred_path = root / "data/derived/model_game_predictions.csv"
    reg_path = root / "data/derived/model_registry.csv"
    if not pred_path.exists():
        return pd.DataFrame()
    canonical = pd.read_csv(pred_path, low_memory=False)
    registry = pd.read_csv(reg_path, low_memory=False) if reg_path.exists() else pd.DataFrame()
    counts = canonical["canonical_model_id"].astype(str).value_counts().to_dict()
    candidates: dict[str, set[str]] = {}
    display: dict[str, str] = {}

    def add(mid: Any, name: Any) -> None:
        if pd.isna(name):
            return
        mid = str(mid); name = str(name).strip()
        if not name:
            return
        display.setdefault(mid, name)
        for alias in _model_aliases(name):
            candidates.setdefault(alias, set()).add(mid)

    if "model_name" in canonical:
        for row in canonical[["canonical_model_id", "model_name"]].drop_duplicates().itertuples(index=False):
            add(row.canonical_model_id, row.model_name)
    for col in ["source_model_name", "raw_model_name", "model_source_name", "source_model_key"]:
        if col in canonical:
            for row in canonical[["canonical_model_id", col]].dropna().drop_duplicates().itertuples(index=False):
                add(row[0], row[1])
    if len(registry) and {"canonical_model_id", "model_name"}.issubset(registry.columns):
        for row in registry[["canonical_model_id", "model_name"]].drop_duplicates().itertuples(index=False):
            add(row.canonical_model_id, row.model_name)
        display.update(dict(zip(
            registry["canonical_model_id"].astype(str),
            registry["model_name"].astype(str),
        )))

    existing_ids = set(canonical["canonical_model_id"].astype(str))
    used_new: set[str] = set()
    rows = []
    for picker in live_pickers:
        hits = {
            mid for alias in _model_aliases(picker)
            for mid in candidates.get(alias, set())
        }
        if hits:
            mid = sorted(hits, key=lambda x: (-int(counts.get(x, 0)), x))[0]
            model_name = display.get(mid, _canonical_live_picker_name(picker))
            is_new = False; reason = "matched_existing_alias"
        else:
            base = f"cfbpicker_{_compact_model(_canonical_live_picker_name(picker))[:60]}"
            mid = base or "cfbpicker_model"; suffix = 2
            while mid in existing_ids or mid in used_new:
                mid = f"{base}_{suffix}"; suffix += 1
            used_new.add(mid)
            model_name = _canonical_live_picker_name(picker)
            is_new = True; reason = "new_cfbpicker_model"
        rows.append({
            "picker_name": picker, "canonical_model_id": mid,
            "model_name": model_name, "is_new_canonical_model": is_new,
            "mapping_reason": reason,
        })
    return pd.DataFrame(rows)


def resolve_mapping(
    root: Path, mapping_file: Path | None, *, season: int,
    picker_file: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    explicit = None
    if mapping_file is not None:
        explicit = mapping_file if mapping_file.is_absolute() else root / mapping_file
    selected = _mapping_frame(
        explicit or root / "data/current/cfbpicker_selected_mapping.csv"
    )
    historical = _mapping_frame(root / "data/derived/cfbpicker_model_mapping.csv")
    live_path = picker_file
    explicit_picker_file = picker_file is not None
    if live_path is not None and not live_path.is_absolute():
        live_path = root / live_path
    if live_path is None:
        live_path = root / f"data/cfbpicker/collectable_pickers_{season}.txt"
    if explicit_picker_file and not live_path.exists():
        raise FileNotFoundError(
            f"Explicit --picker-file was not found: {live_path}. "
            "Refusing to silently fall back to the full picker list."
        )
    live_pickers = _picker_names(live_path) if live_path.exists() else []

    selected_ids = set(selected.get("canonical_model_id", pd.Series(dtype=str)).astype(str))
    selected_keys = set(selected.get("picker_name", pd.Series(dtype=str)).astype(str).map(safe_key))
    if live_pickers:
        mapping = _dynamic_live_mapping(root, live_pickers)
        if mapping.empty:
            known = pd.concat([selected, historical], ignore_index=True, sort=False)
            by_picker = {safe_key(r.picker_name): r for r in known.itertuples(index=False)}
            rows = []
            for picker in live_pickers:
                prior = by_picker.get(safe_key(picker))
                if prior is not None:
                    rows.append({
                        "picker_name": picker,
                        "canonical_model_id": str(prior.canonical_model_id),
                        "model_name": str(prior.model_name),
                        "is_new_canonical_model": False,
                        "mapping_reason": "matched_existing_picker_mapping",
                    })
                else:
                    rows.append({
                        "picker_name": picker,
                        "canonical_model_id": f"cfbpicker_{safe_key(picker)}",
                        "model_name": picker,
                        "is_new_canonical_model": True,
                        "mapping_reason": "new_live_cfbpicker_model",
                    })
            mapping = pd.DataFrame(rows)
    else:
        mapping = selected.copy() if len(selected) else historical.copy()
        if mapping.empty:
            raise FileNotFoundError("No usable CFB Picker mapping or live picker file found.")

    selected_by_picker = {safe_key(r.picker_name): r for r in selected.itertuples(index=False)}
    for idx, row in mapping.iterrows():
        prior = selected_by_picker.get(safe_key(row["picker_name"]))
        if prior is not None:
            mapping.at[idx, "canonical_model_id"] = str(prior.canonical_model_id)
            mapping.at[idx, "model_name"] = str(prior.model_name)
            mapping.at[idx, "is_new_canonical_model"] = False
            mapping.at[idx, "mapping_reason"] = "selected_mapping_exact_picker"
    if "is_new_canonical_model" not in mapping:
        mapping["is_new_canonical_model"] = False
    if "mapping_reason" not in mapping:
        mapping["mapping_reason"] = "existing_mapping"
    mapping["strategy_selected"] = [
        str(mid) in selected_ids or safe_key(picker) in selected_keys
        for mid, picker in zip(mapping["canonical_model_id"], mapping["picker_name"])
    ]
    mapping = mapping.drop_duplicates(
        ["picker_name", "canonical_model_id"], keep="first"
    ).sort_values(["strategy_selected", "picker_name"], ascending=[False, True])

    derived = root / "data/derived"; derived.mkdir(parents=True, exist_ok=True)
    live_mapping_path = derived / f"cfbpicker_live_model_mapping_{season}.csv"
    mapping.to_csv(live_mapping_path, index=False)
    mapping.to_csv(derived / "cfbpicker_live_model_mapping.csv", index=False)
    live_ids = set(mapping["canonical_model_id"].astype(str))
    live_keys = set(mapping["picker_name"].astype(str).map(safe_key))
    selected_not_live = selected[
        ~selected["canonical_model_id"].astype(str).isin(live_ids)
        & ~selected["picker_name"].astype(str).map(safe_key).isin(live_keys)
    ] if len(selected) else selected
    new_mask = mapping["is_new_canonical_model"].astype(str).str.lower().isin({"true", "1", "yes"})
    meta = {
        "picker_file": str(live_path), "picker_file_found": bool(live_path.exists()),
        "live_picker_values": int(len(live_pickers)), "models_resolved": int(len(mapping)),
        "strategy_models_requested": int(len(selected)),
        "strategy_models_live": int(mapping["strategy_selected"].sum()),
        "strategy_models_not_live": selected_not_live[
            ["canonical_model_id", "model_name", "picker_name"]
        ].to_dict("records") if len(selected_not_live) else [],
        "new_live_canonical_models": mapping.loc[
            new_mask, ["canonical_model_id", "model_name", "picker_name"]
        ].to_dict("records"),
        "mapping_output": str(live_mapping_path.relative_to(root)),
    }
    return mapping[[
        "canonical_model_id", "model_name", "picker_name", "strategy_selected",
        "is_new_canonical_model", "mapping_reason",
    ]].reset_index(drop=True), meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--mapping-file", type=Path, default=None)
    ap.add_argument("--picker-file", type=Path, default=None)
    ap.add_argument("--mapping-only", action="store_true")
    ap.add_argument(
        "--week-filter", default=None,
        help="Override Tableau's Week filter value (for example 1-2).",
    )
    ap.add_argument(
        "--completion-filter", default=None,
        help="Diagnostic override for Tableau's Game Complete filter.",
    )
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--system-chrome", action="store_true")
    ap.add_argument("--wait-seconds", type=int, default=4)
    ap.add_argument("--delay-seconds", type=float, default=0.25)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--quick", action="store_true")  # compatibility
    ap.add_argument("--force-download", action="store_true")  # compatibility
    ap.add_argument("--no-error-screenshots", action="store_true")
    args = ap.parse_args()
    root = args.root.expanduser().resolve()
    history = load_history_module(root)
    try:
        mapping, mapping_meta = resolve_mapping(
            root, args.mapping_file, season=args.season,
            picker_file=args.picker_file,
        )
    except Exception as exc:
        print(json.dumps({"usable": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2
    if args.mapping_only:
        print(json.dumps({
            "collector_version": COLLECTOR_VERSION,
            "season": int(args.season), "week": int(args.week),
            **mapping_meta, "mapping": mapping.to_dict("records"),
        }, indent=2))
        return 0
    master = load_master_slate(root)
    output_path = root / "data/current/cfbpicker_current_long.csv"
    status_path = root / "data/derived/cfbpicker_current_source_status.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    if master.empty:
        status = {
            "collector_version": COLLECTOR_VERSION,
            "season": int(args.season),
            "week": int(args.week),
            "rows": 0,
            "models_ok": 0,
            "models_requested": int(len(mapping)),
            "usable": False,
            "error": (
                "PredictionTracker current slate is missing or invalid at "
                "data/current/ncaapredictions.csv; refusing to label an "
                "unmatched Tableau season page as the requested week."
            ),
        }
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        print(json.dumps(status, indent=2))
        return 2
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        status = {"collector_version": COLLECTOR_VERSION, "season": args.season, "week": args.week, "rows": 0, "models_ok": 0, "models_requested": int(len(mapping)), "usable": False, "error": "Playwright is not installed."}
        status_path.write_text(json.dumps(status, indent=2))
        print(json.dumps(status, indent=2))
        return 2
    frames, records = [], []
    with tempfile.TemporaryDirectory(prefix="cfbpicker_current_api_") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "index.html").write_text(embedding_html(), encoding="utf-8")
        with local_server(tmp_path) as server_url:
            with sync_playwright() as playwright:
                launch = {"headless": not args.headed}
                if args.system_chrome:
                    launch["channel"] = "chrome"
                browser = playwright.chromium.launch(**launch)
                context = browser.new_context(viewport={"width": 1750, "height": 1300}, locale="en-US")
                for row in mapping.itertuples(index=False):
                    picker = str(row.picker_name)
                    print(f"Current API season={args.season} week={args.week} picker={picker}")
                    data, record = scrape_picker(
                        context, server_url, root, history, master,
                        season=args.season, week=args.week, picker=picker,
                        canonical_model_id=str(row.canonical_model_id), model_name=str(row.model_name),
                        wait_seconds=args.wait_seconds, screenshot_errors=not args.no_error_screenshots,
                        week_filter_override=args.week_filter,
                        completion_filter_override=args.completion_filter,
                    )
                    record["strategy_selected"] = bool(row.strategy_selected)
                    records.append(record)
                    if len(data):
                        frames.append(data)
                    print(
                        f"  {record['status']}: "
                        f"{record['rows_extracted']}/{record['rows_detected']}; "
                        f"parsed={record.get('rows_parsed_before_master_slate', 0)}; "
                        f"failed={record.get('rows_failed', 0)}"
                    )
                    if record.get("message"):
                        print(f"  {record['message']}", file=sys.stderr)
                    if args.delay_seconds:
                        time.sleep(args.delay_seconds)
                browser.close()
    if frames:
        current = pd.concat(frames, ignore_index=True).drop_duplicates(["season", "week", "away", "home", "canonical_model_id"])
        current["source_table"] = "tableau_embedding_api_tooltip"
        current["orientation_method"] = "explicit_home_margin_tooltip"
        current.to_csv(output_path, index=False)
    else:
        current = pd.DataFrame()
        if output_path.exists():
            output_path.unlink()
    status = {
        "collector_version": COLLECTOR_VERSION, "season": int(args.season), "week": int(args.week),
        "rows": int(len(current)), "models_ok": int(sum(r["status"] == "ok" for r in records)),
        "models_no_data": int(sum(r["status"] == "no_data" for r in records)),
        "models_requested": int(len(mapping)), "usable": bool(len(current) > 0),
        "output": str(output_path.relative_to(root)), "mapping": mapping_meta,
        "records": records,
    }
    status_path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "collector_version": COLLECTOR_VERSION, "season": status["season"], "week": status["week"],
        "rows": status["rows"], "models_ok": status["models_ok"], "models_no_data": status["models_no_data"],
        "models_requested": status["models_requested"], "usable": status["usable"],
        "mapping": mapping_meta,
        "status": str(status_path.relative_to(root)),
        "failed_models": [{"picker": r["picker"], "status": r["status"], "message": r.get("message")} for r in records if r["status"] == "error"],
    }, indent=2))
    if args.strict and any(
        r["status"] == "error" and r.get("strategy_selected")
        for r in records
    ):
        return 2
    return 0 if len(current) else 2


if __name__ == "__main__":
    raise SystemExit(main())
