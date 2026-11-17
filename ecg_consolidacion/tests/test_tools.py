import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import check_frozen  # noqa: E402
import check_reserved_paths  # noqa: E402
import registry  # noqa: E402
from engine.contracts import ecg_input  # noqa: E402

SHA = "a" * 64


# ----------------------------------------------------------------- registry
def _reg():
    return registry.load(registry.MODELS), registry.load(registry.DATASETS)


def test_real_registry_is_valid():
    m, d = _reg()
    assert registry.validate(m, d) == []


def test_no_model_is_externally_validated_today():
    m, _ = _reg()
    assert all(x["status"] != "VE" for x in m["models"])


def test_ve_without_evidence_is_rejected():
    m, d = _reg()
    m = copy.deepcopy(m)
    m["models"][0]["status"] = "VE"
    assert any("VE sin external_validation_evidence" in e for e in registry.validate(m, d))


def test_ve_on_consumed_dataset_is_rejected():
    m, d = _reg()
    m = copy.deepcopy(m)
    m["models"][0]["status"] = "VE"
    m["models"][0]["external_validation_evidence"] = {
        "protocol_file": "protocols/x.yaml", "protocol_sha256": SHA,
        "dataset": "ptb_qt400", "dataset_status_before": "CONSUMED"}
    assert any("FRESH" in e for e in registry.validate(m, d))


def test_unknown_dataset_reference_is_rejected():
    m, d = _reg()
    m = copy.deepcopy(m)
    m["models"][0]["data"]["trained_on"].append("dataset_inventado")
    assert any("dataset_inventado" in e for e in registry.validate(m, d))


def test_malformed_sha_is_rejected():
    m, d = _reg()
    m = copy.deepcopy(m)
    for mod in m["models"]:
        for a in mod.get("artifacts") or []:
            if isinstance(a, dict) and a.get("sha256"):
                a["sha256"] = "XYZ"
                break
        else:
            continue
        break
    assert any("mal formado" in e for e in registry.validate(m, d))


def test_dataset_reference_resolution():
    ids = ["ptb_xl", "ludb", "code15", "code_test"]
    assert registry.resolve_dataset("ptb_xl_folds_1_8", ids) == "ptb_xl"
    assert registry.resolve_dataset("code15_test_v12", ids) == "code15"
    assert registry.resolve_dataset("code_test", ids) == "code_test"
    assert registry.resolve_dataset("nada", ids) is None


def test_frozen_models_have_lock_coverage():
    lock = json.loads((ROOT / "frozen" / "FROZEN.lock.json").read_text())
    m, _ = _reg()
    frozen = {x["id"] for x in m["models"] if x.get("frozen")}
    assert frozen == set(lock["models"])


# ------------------------------------------------------------------- frozen
def test_frozen_lock_matches_repository():
    expected = json.loads((ROOT / "frozen" / "FROZEN.lock.json").read_text())
    assert check_frozen.diff(expected, check_frozen.build_lock()) == []


def test_frozen_detects_sha_change():
    expected = json.loads((ROOT / "frozen" / "FROZEN.lock.json").read_text())
    actual = copy.deepcopy(expected)
    mid = next(k for k, v in actual["models"].items() if any(v.values()))
    art = next(k for k, v in actual["models"][mid].items() if v)
    actual["models"][mid][art] = SHA
    assert check_frozen.diff(expected, actual)


def test_frozen_detects_protected_file_change():
    expected = json.loads((ROOT / "frozen" / "FROZEN.lock.json").read_text())
    actual = copy.deepcopy(expected)
    f = next(iter(actual["files"]))
    actual["files"][f] = SHA
    assert any("archivo protegido" in p for p in check_frozen.diff(expected, actual))


def test_split_hashes_match_registry():
    _, d = _reg()
    ludb = next(x for x in d["datasets"] if x["id"] == "ludb")
    for split in ludb["splits"].values():
        if split.get("file"):
            assert check_frozen.sha256_file(ROOT / split["file"]) == split["sha256"]


# ---------------------------------------------------------------- reserved
@pytest.fixture(scope="module")
def rules():
    return check_reserved_paths.compile_rules()


def test_reserved_detects_ptb400(rules):
    assert "ptb_qt400" in check_reserved_paths.scan_text("p = '/data/PTB400/records'", rules)


def test_reserved_detects_trihead_test_split(rules):
    code = "s = json.load(open('SPLIT_PATIENTS_LOCKED.json'))\nids = s['subjects']['test']"
    assert "ludb_trihead_test" in check_reserved_paths.scan_text(code, rules)


def test_reserved_allows_trihead_val(rules):
    code = "s = json.load(open('SPLIT_PATIENTS_LOCKED.json'))\nids = s['subjects']['val']"
    assert check_reserved_paths.scan_text(code, rules) == []


def test_reserved_approval_comment(rules):
    code = "# reserved-access-approved: heedb PR#12\nroot = 'HEEDB/ECG'"
    assert check_reserved_paths.scan_text(code, rules) == []


def test_repository_code_is_clean():
    assert check_reserved_paths.main([]) == 0


# ---------------------------------------------------------------- contract
def _ok_signal():
    t = np.linspace(0, 10, 5000)
    return np.tile(np.sin(2 * np.pi * 1.2 * t), (12, 1)).astype(np.float32)


def _kw(**over):
    kw = dict(fs_hz=500, units="mV", leads=ecg_input.LEAD_ORDER, source="TEST:1", source_sha256=SHA)
    kw.update(over)
    return kw


def test_contract_accepts_canonical():
    ecg = ecg_input.validate(_ok_signal(), **_kw())
    assert ecg.signal.shape == (12, 5000)


@pytest.mark.parametrize("over,msg", [
    (dict(fs_hz=100), "fs_hz"),
    (dict(units="uV"), "units"),
    (dict(leads=tuple(reversed(ecg_input.LEAD_ORDER))), "orden"),
    (dict(source_sha256="abc"), "SHA-256"),
])
def test_contract_rejects_metadata(over, msg):
    with pytest.raises(ecg_input.ContractError, match=msg):
        ecg_input.validate(_ok_signal(), **_kw(**over))


def test_contract_rejects_microvolts():
    with pytest.raises(ecg_input.ContractError, match="µV"):
        ecg_input.validate(_ok_signal() * 1000, **_kw())


def test_contract_rejects_nan_and_shape():
    s = _ok_signal()
    s[3, 10] = np.nan
    with pytest.raises(ecg_input.ContractError, match="NaN"):
        ecg_input.validate(s, **_kw())
    with pytest.raises(ecg_input.ContractError, match="shape"):
        ecg_input.validate(_ok_signal()[:, :1000], **_kw())


def test_contract_flags_flat_lead():
    s = _ok_signal()
    s[6] = 0.0
    ecg = ecg_input.validate(s, **_kw())
    assert any("V1" in n for n in ecg.notes)
