#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MEDCALC EKG UNIFICADO V52: research-only unified inference, no ECG downloads.

The frozen components are never overwritten:
- V51 DIAG_BEST: PTB-XL superclasses at 100 Hz.
- V51 WAVE_BEST: LUDB P/QRS/T masks at 500 Hz.
- R27: exactly 35 research probabilities, no binary diagnostic decisions.
- Historical H211: optional legacy digitized signal ZIP measurement path.
- U-Net: audited, but images cannot be measured without a verified V07 router
  and pixel/time/amplitude calibration; fail closed on image input.

The V52 folder contains ONE command/JSON report. Existing component artefacts
remain in their original, read-only folders. No external datasets are fetched.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import traceback

ROOT = Path(os.environ.get("MEDCALC_EKG_ROOT", "/content/drive/MyDrive/MEDCALC"))
BASE = ROOT / "MEDCALC_EKG_V50_NEURAL"
V51 = BASE / "RESULTADOS" / "ENTRENAMIENTO_V51_SIN_DESCARGAS"
R27 = ROOT / "MEDCALC_R27_BACKEND_DEPLOY"
DIG = ROOT / "MEDCALC_OPEN_ECG_DIGITIZER"
OUT = ROOT / "MEDCALC_EKG_UNIFICADO_V52" / "RESULTADOS"
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
SUPER = ("NORM", "MI", "STTC", "CD", "HYP")
RESEARCH_ONLY = "CLINICAL_DEPLOYMENT_BLOCKED"


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                return h.hexdigest()
            h.update(b)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False,
                              default=lambda x: float(x) if hasattr(x, "item") else str(x)),
                   encoding="utf-8")
    tmp.replace(path)


def audit():
    items = {
        "v51_classifier": V51 / "DIAG_BEST.pt",
        "v51_delineator": V51 / "WAVE_BEST.pt",
        "v51_wave_thresholds": V51 / "WAVE_SELECTION.json",
        "v51_source": BASE / "medcalc_ekg_v51_FOLD9_FIXED.py",
        "v51_source_fallback": BASE / "medcalc_ekg_v51_SIN_DESCARGAS.py",
        "v51_core": BASE / "medcalc_ekg_v51_12k.py",
        "r27_backend": R27 / "api.py",
        "r27_gate": R27 / "MEDCALC_R27_PROBABILITY_ONLY_GATE.py",
        "legacy_h211": R27 / "compat" / "MEDCALC" / "_r27_root_sources" / "MEAS_H211.py",
        "unet_source": DIG / "source" / "unet.py",
        "unet_extractor": DIG / "source" / "signal_extractor.py",
        "unet_weights": DIG / "weights" / "unet_weights_07072025.pt",
    }
    status = {k: {"exists": v.is_file(), "size": v.stat().st_size if v.is_file() else None,
                  "path": str(v)} for k, v in items.items()}
    mandatory = ("v51_classifier", "v51_delineator", "v51_wave_thresholds", "v51_core")
    return {"version": "V52_UNIFIED", "generated_utc": utc(),
            "status": status, "ready_neural": all(status[k]["exists"] for k in mandatory),
            "ready_r27": status["r27_backend"]["exists"] and status["r27_gate"]["exists"],
            "ready_legacy_measure_zip": status["legacy_h211"]["exists"],
            "unet_assets_present": all(status[k]["exists"] for k in ("unet_source", "unet_extractor", "unet_weights")),
            "image_to_12lead_calibrated_runtime_verified": False,
            "r27_binary_diagnoses_authorized": False,
            "clinical_state": RESEARCH_ONLY}


def import_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, filename)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to import: " + str(filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def waveform(record_stem):
    import numpy as np
    import wfdb
    from scipy.signal import resample_poly
    stem = str(record_stem)
    if stem.endswith(".hea") or stem.endswith(".dat"):
        stem = stem[:-4]
    record = wfdb.rdrecord(stem)
    if record.p_signal is None:
        raise ValueError("The ECG must provide physical signals calibrated in mV.")
    fs = float(record.fs)
    if fs < 100 or fs > 2000:
        raise ValueError("Sampling frequency outside verified interval 100-2000 Hz")
    names = [re.sub(r"[^A-Za-z0-9]", "", name).upper() for name in record.sig_name]
    desired = [re.sub(r"[^A-Za-z0-9]", "", name).upper() for name in LEADS]
    if len(set(names)) != len(names) or any(x not in names for x in desired):
        raise ValueError("All 12 canonical leads required; do not infer lead order")
    if any(str(u).strip().lower() != "mv" for u in record.units):
        raise ValueError("Physical units must be mV in every lead; explicit conversion required")
    s = np.asarray(record.p_signal[:, [names.index(x) for x in desired]], dtype=np.float64)
    if s.ndim != 2 or len(s) / fs < 9.95:
        raise ValueError("At least 10 seconds of simultaneous ECG signals required")
    s = s[:int(round(10.0 * fs))]
    if not np.isfinite(s).all() or np.max(np.abs(s)) > 30 or s.shape[1] != 12:
        raise ValueError("Invalid/nonfinite amplitude or lead count; mV scaling required")
    def down_or_resample(newfs):
        f = Fraction(newfs / fs).limit_denominator(2000)
        x = resample_poly(s, f.numerator, f.denominator, axis=0)
        if len(x) != newfs * 10:
            raise ValueError("ECG resampling did not preserve 10-second duration")
        return np.asarray(x, dtype=np.float32)
    lr = down_or_resample(100)
    hr = down_or_resample(500) if fs >= 500 else None
    return s, lr, hr, fs


def neural_models(low100, high500):
    import numpy as np
    import torch
    if not V51.joinpath("DIAG_BEST.pt").exists():
        raise FileNotFoundError("Missing V51 DIAG_BEST.pt")
    sys.path.insert(0, str(BASE))
    mod_path = BASE / "medcalc_ekg_v51_FOLD9_FIXED.py"
    if not mod_path.is_file():
        mod_path = BASE / "medcalc_ekg_v51_SIN_DESCARGAS.py"
    mod = import_module(mod_path, "_medcalc_v52_v51")
    core = mod.core
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    o = torch.load(V51 / "DIAG_BEST.pt", map_location="cpu", weights_only=True)
    m = o["meta"]
    if int(m["fs"]) != 100 or tuple(m["labels"]) != SUPER:
        raise RuntimeError("V51 diagnostic checkpoint incompatible with input contract")
    net = mod.Classifier100(base=int(m["base"]), n_scp=len(m["scp_vocab"]))
    net.load_state_dict(o["state_dict"], strict=True)
    net.to(dev).eval()
    x = torch.from_numpy(np.ascontiguousarray(np.clip(low100.T / 2.0, -5.0, 5.0))).float()[None].to(dev)
    with torch.inference_mode():
        coarse, codes = net(x)
    diagnostic = {"source": "V51_DIAG_BEST_100HZ",
                  "checkpoint_sha256": sha(V51 / "DIAG_BEST.pt"),
                  "five_superclass_probabilities": dict(zip(SUPER, torch.sigmoid(coarse)[0].cpu().tolist())),
                  "scp_probabilities": dict(zip(m["scp_vocab"], torch.sigmoid(codes)[0].cpu().tolist())),
                  "validated_internal_only": True, "binary_diagnosis": None,
                  "clinical_state": RESEARCH_ONLY}
    if high500 is None:
        return diagnostic, {"status": "NOT_ASSESSABLE_INPUT_BELOW_500HZ",
                            "reason": "Cannot obtain 500 Hz morphological resolution by upsampling"}
    w = torch.load(V51 / "WAVE_BEST.pt", map_location="cpu", weights_only=True)
    wm = w["meta"]
    if int(wm["fs"]) != 500 or tuple(wm["wave_labels"]) != ("BACKGROUND", "P", "QRS", "T"):
        raise RuntimeError("V51 wave checkpoint incompatible")
    wave = core.ECGMultiTask(base=int(wm["base"]), n_scp=0)
    wave.load_state_dict(w["state_dict"], strict=True)
    wave.to(dev).eval()
    y = torch.from_numpy(np.ascontiguousarray(high500.T / 2.0)).float()[None].to(dev)
    with torch.inference_mode():
        _, logits = wave(y)
        prob = torch.softmax(logits, dim=2)[0].cpu().numpy()
    thresholds = json.loads((V51 / "WAVE_SELECTION.json").read_text(encoding="utf-8"))["postprocess"]
    pt = float(thresholds["p_threshold"])
    tt = float(thresholds["t_threshold"])
    if not (0.1 <= pt <= 0.99 and 0.1 <= tt <= 0.99):
        raise ValueError("Unacceptable P/T thresholds")
    seg = prob.argmax(axis=1).astype(np.uint8)
    seg[(seg == 1) & (prob[:, 1, :] < pt)] = 0
    seg[(seg == 3) & (prob[:, 3, :] < tt)] = 0
    return diagnostic, {"status": "PREDICTED_RESEARCH_ONLY", "sample_rate_hz": 500,
                        "model_sha256": sha(V51 / "WAVE_BEST.pt"),
                        "mask": seg, "thresholds": {"P": pt, "T": tt}}


def blocks(mask, cls, minimum_ms, maximum_ms, fs=500):
    import numpy as np
    idx = np.flatnonzero(mask == cls)
    if not len(idx):
        return []
    splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    return [(int(v[0]), int(v[-1] + 1)) for v in splits
            if minimum_ms <= (len(v) / fs * 1000) <= maximum_ms]


def robust_med(vals):
    import numpy as np
    return round(float(np.median(vals)), 2) if vals else None


def measure_from_masks(high, masks):
    """Research interval estimates, not a substitute for validated V23.H211.
    Reject ambiguous intervals rather than return fabricated values.
    """
    import numpy as np
    fs = 500
    ii = masks[1]
    qs = blocks(ii, 2, 40, 240)
    ps = blocks(ii, 1, 30, 220)
    ts = blocks(ii, 3, 60, 550)
    rr = [1000.0 * (qs[i][0] - qs[i-1][0]) / fs for i in range(1, len(qs))]
    rr = [x for x in rr if 260 <= x <= 3000]
    rr_cv = float(np.std(rr) / np.mean(rr)) if len(rr) >= 2 else None
    pr, qrs, qt = [], [], []
    for i, (a, b) in enumerate(qs):
        width = 1000.0 * (b-a)/fs
        qrs.append(width)
        prev = [p for p in ps if 70 <= 1000*(a-p[0])/fs <= 400 and p[1] <= a]
        if prev:
            pr.append(1000.0*(a-max(prev, key=lambda v:v[0])[0])/fs)
        nxt = [t for t in ts if t[0] >= b and 150 <= 1000*(t[1]-a)/fs <= 750]
        if nxt:
            qt.append(1000.0*(min(nxt, key=lambda v:v[0])[1]-a)/fs)
    n = len(qs)
    rr_ms = robust_med(rr) if n >= 3 and len(rr) >= 2 else None
    qrs_ms = robust_med(qrs) if n >= 2 else None
    pr_ms = robust_med(pr) if len(pr) >= max(2, n//2) else None
    qt_ms = robust_med(qt) if len(qt) >= max(2, n//2) else None
    regular = rr_cv is not None and rr_cv < 0.12
    qtc = {}
    if regular and rr_ms and qt_ms and 300 < rr_ms < 2000:
        rr_s = rr_ms/1000.0
        bpm = 60000/rr_ms
        qtc = {"bazett_ms": round(qt_ms/math.sqrt(rr_s),2),
               "fridericia_ms": round(qt_ms/(rr_s**(1/3)),2),
               "framingham_ms": round(qt_ms+154*(1-rr_s),2),
               "hodges_ms": round(qt_ms+1.75*(bpm-60),2)}
    else:
        qtc = {"bazett_ms": None, "fridericia_ms": None, "framingham_ms": None,
               "hodges_ms": None}
    # Net frontal QRS areas, proxy only; clinical axis requires separate validation.
    axis = None
    if n >= 2 and high is not None:
        areas = []
        for lead in (0, 5):
            v = high[:,lead]
            xs = []
            for a,b in qs:
                if a < 20 or b+20 >= len(v): continue
                baseline = float(np.median(v[a-20:a]))
                xs.append(float(np.trapezoid(v[a:b]-baseline,dx=1/fs)))
            areas.append(robust_med(xs))
        if all(v is not None for v in areas) and abs(areas[0])+abs(areas[1])>1e-5:
            axis = round(float(np.degrees(np.arctan2(areas[1],areas[0]))),1)
    qc = "PARTIAL_RESEARCH_ESTIMATE" if n >= 2 else "NOT_ASSESSABLE"
    return {"status": qc, "algorithm": "V51 mask-based beat pairing, lead II",
            "qrs_events_ii": n, "rr_series_ms": [round(x,2) for x in rr],
            "rr_cv": round(rr_cv,3) if rr_cv is not None else None,
            "heart_rate_bpm": round(60000/rr_ms,1) if rr_ms else None,
            "rr_median_ms": rr_ms, "pr_ms": pr_ms, "qrs_ms": qrs_ms,
            "qt_ms": qt_ms, "qtc": qtc,
            "axis_qrs_proxy_degrees": axis,
            "qc_warnings": ["Lead-II mask estimates require adjudicated interval validation",
                             "QTc suppressed if irregular RR or uncertain intervals",
                             "Axis is exploratory net-area proxy; not a clinical axis"],
            "clinical_state": RESEARCH_ONLY}


def h211_from_legacy_zip(zip_path):
    path = R27 / "compat" / "MEDCALC" / "_r27_root_sources" / "MEAS_H211.py"
    if not path.is_file():
        return {"status":"NOT_AVAILABLE","reason":"Historical MEAS_H211.py missing"}
    try:
        h = import_module(path, "_medcalc_h211_v52")
        result = h.measure_zip(str(zip_path))
        return {"status":result.get("measurement_qc","UNKNOWN"),"algorithm":"V23.HYBRID2.1.1",
                "measurements":h.compact_result(result),
                "source":"original historical engine, source ZIP",
                "clinical_state":RESEARCH_ONLY}
    except Exception as exc:
        return {"status":"FAILED_CLOSED","reason":f"{type(exc).__name__}: {exc}"}


def legacy35(hr500, lr100, age, sex):
    if hr500 is None:
        return {"status":"NOT_ASSESSABLE","reason":"500-Hz physical waveform required for the R27 exact runtime"}
    if age is None or sex is None:
        return {"status":"NOT_ASSESSABLE","reason":"R27 requires explicit numeric age and legacy sex code 0/1"}
    if not (0 <= age <= 120 and sex in (0,1)):
        raise ValueError("age 0..120 and sex 0/1 required for frozen R27 runtime")
    if not (R27/"api.py").is_file():
        return {"status":"NOT_AVAILABLE","reason":"Exact R27 backend absent; no substitute 35-module predictions"}
    import wfdb
    with tempfile.TemporaryDirectory(prefix="v52_r27_") as t:
        t=Path(t); ptb=t/"ptb-xl-1.0.3"
        for sub, sample_rate, data in (("hr",500,hr500),("lr",100,lr100)):
            folder=ptb/"request"/sub
            folder.mkdir(parents=True,exist_ok=True)
            wfdb.wrsamp(record_name="ecg_"+sub,fs=sample_rate,
                         units=["mV"]*12,sig_name=list(LEADS),
                         p_signal=data.astype("float64"),
                         fmt=["16"]*12,write_dir=str(folder))
        api=import_module(R27/"api.py","_medcalc_r27_v52")
        meta=api._prepare_database_and_meta(ptb,age=age,sex=sex,
                                            filename_hr="request/hr/ecg_hr",
                                            filename_lr="request/lr/ecg_lr")
        result=api._run_exact_runtime(ptb_root=ptb,meta_path=meta,request_root=t)
    api.GATE.assert_probability_only_payload(result)
    if result.get("module_n")!=35:
        raise RuntimeError("R27 did not return exactly 35 research modules")
    return {"status":"PROBABILITY_ONLY","source":"Frozen R27 exact research runtime",
            "output":result,"clinical_state":RESEARCH_ONLY}


def analyze(args):
    import numpy as np
    pre=audit()
    if not pre["ready_neural"]:
        raise RuntimeError("Neural assets missing; run --audit first")
    if args.image:
        raise RuntimeError("IMAGE_PATH_BLOCKED: U-Net weights exist, but validated V07 image-to-12-lead layout/calibration is not yet recoverable. Do not substitute pixels for mV. Supply a calibrated WFDB record.")
    if not args.wfdb:
        raise ValueError("--wfdb path to calibrated 12-lead 10-second WFDB record required")
    raw, lo, hi, fs=waveform(args.wfdb)
    diag, delineation=neural_models(lo,hi)
    if delineation.get("status")=="PREDICTED_RESEARCH_ONLY":
        seg=delineation.pop("mask")
        metrics=measure_from_masks(hi,seg)
        wave_summary={**delineation,"wave_class_count":4,"detected_qrs_lead_ii":metrics["qrs_events_ii"]}
    else:
        metrics={"status":"NOT_ASSESSABLE","reason":"500Hz input required for morphological measurements"}
        wave_summary=delineation
    h211=h211_from_legacy_zip(args.legacy_measure_zip) if args.legacy_measure_zip else {
        "status":"NOT_RUN","reason":"Historical V23.H211 requires its calibrated legacy ZIP format"}
    try:
        thirtyfive=legacy35(hi,lo,args.age,args.sex) if not args.skip_r27 else {
            "status":"SKIPPED_BY_REQUEST"}
    except Exception as exc:
        thirtyfive={"status":"FAILED_CLOSED","reason":f"{type(exc).__name__}: {exc}",
                    "traceback_tail":traceback.format_exc()[-1200:]}
    dest=Path(args.output) if args.output else OUT/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_ECG.json")
    data={"version":"V52_UNIFIED_RESEARCH","created_utc":utc(),
          "input":{"source_type":"WFDB_PHYSICAL_MV","frequency_hz":fs,"sample_count":len(raw),
                   "leads":list(LEADS),"patient_identifiers_exported":False},
          "digitalizer":{"status":"BYPASSED_DIGITAL_INPUT","unet_assets_present":pre["unet_assets_present"]},
          "delineation":wave_summary,"measurements":metrics,
          "historical_measurement_engine":h211,
          "diagnosis":{"v51":diag,"legacy_35_r27":thirtyfive},
          "warnings":["Not validated for clinical decision-making",
                      "Historical 35 modules: probability-only if available",
                      "V51 trained on PTB-XL/LUDB; no independent external clinical validation",
                      "No shared end-to-end retraining has been conducted"],
          "clinical_state":RESEARCH_ONLY}
    write_json(dest,data)
    print(json.dumps({"status":"REPORT_WRITTEN","path":str(dest),
                      "legacy_35_status":thirtyfive.get("status"),
                      "measurements_status":metrics.get("status"),
                      "clinical_state":RESEARCH_ONLY},ensure_ascii=False))
    return data


def smoke():
    import numpy as np
    fs=500
    m=np.zeros((12,5000),dtype=np.uint8)
    for q in (400,900,1400,1900,2400,2900,3400,3900,4400):
        m[1,q:q+45]=2
        m[1,q-95:q-55]=1
        m[1,q+95:q+210]=3
    measured=measure_from_masks(np.zeros((5000,12),dtype=np.float32),m)
    assert measured["heart_rate_bpm"]==60.0, measured
    assert 150 <= measured["pr_ms"] <= 250
    assert measured["qrs_ms"]==90.0
    assert measured["qtc"]["fridericia_ms"] is not None
    assert audit()["clinical_state"]==RESEARCH_ONLY
    print("SMOKE PASS (synthetic masks only; not a clinical validation).")


def main():
    p=argparse.ArgumentParser(description="MEDCALC EKG V52: one read-only multimotor interface")
    p.add_argument("command",choices=("audit","analyze","smoke"))
    p.add_argument("--wfdb",help="Physical 12-lead WFDB stem (.hea/.dat), 10 sec")
    p.add_argument("--image",help="Image path: deliberately blocked pending calibrated V07 router")
    p.add_argument("--legacy-measure-zip",help="Optional V23.H211-compatible serialized signal ZIP")
    p.add_argument("--age",type=float,help="Required by frozen R27 runtime")
    p.add_argument("--sex",type=int,choices=(0,1),help="R27 legacy encoded sex 0/1")
    p.add_argument("--skip-r27",action="store_true")
    p.add_argument("--output",help="Single combined JSON output path")
    args=p.parse_args()
    if args.command=="audit":
        result=audit()
        print(json.dumps(result,indent=2,ensure_ascii=False))
        write_json(OUT/"INTEGRATION_AUDIT.json",result)
    elif args.command=="smoke":
        smoke()
    else:
        analyze(args)


if __name__=="__main__":
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        print("\\n[V52 ERROR INTERNO - TRACEBACK COMPLETO]\\n" + error, file=sys.stderr, flush=True)
        try:
            write_json(OUT / "LAST_FAILURE_V52.json",
                       {"version": "V52_UNIFIED", "utc": utc(),
                        "traceback": error, "clinical_state": RESEARCH_ONLY})
            print("[V52] Error guardado en " + str(OUT / "LAST_FAILURE_V52.json"),
                  file=sys.stderr, flush=True)
        except Exception as logging_exc:
            print("[V52] No se pudo guardar el log: " + repr(logging_exc),
                  file=sys.stderr, flush=True)
        raise SystemExit(1)
