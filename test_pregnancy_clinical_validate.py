#!/usr/bin/env python3
"""Regression tests for MEDCALC pregnancy clinical CSV safety gate."""
import csv
import tempfile
import unittest
from pathlib import Path

from MEDCALC_PREGNANCY_CLINICAL_VALIDATE import (
    DEFAULT_CSV,
    REQUIRED_COLUMNS,
    audit,
    generic_or_invalid_url,
)

def valid_row():
    return dict.fromkeys(REQUIRED_COLUMNS, "") | {
        "med_id": "MED-0001",
        "generic_name": "TEST DRUG",
        "status": "REVIEWED",
        "recommendation": "USE_WITH_CAUTION",
        "risk_summary": "Risk with clinical context",
        "clinical_considerations": "Indication-specific management",
        "pregnancy_indication_note": "Only if indicated",
        "fetal_neonatal_risk": "Data limited",
        "evidence_level": "LIMITED",
        "source_authority": "Official specific label",
        "source_url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=00000000-0000-0000-0000-000000000001",
        "reviewed_at": "2026-10-17",
    }

def audit_rows(rows):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pregnancy.csv"
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return audit(path)


class PregnancyClinicalValidatorTests(unittest.TestCase):
    def test_repository_catalogue(self):
        report = audit(DEFAULT_CSV)
        self.assertEqual(report["errors"], [], report["errors"])
        self.assertEqual(report["counts"]["rows"], 106)
        self.assertEqual(report["counts"]["unique_med_ids"], 106)
        self.assertEqual(report["counts"]["REVIEWED"], 82)
        self.assertEqual(report["counts"]["PENDING_REVIEW"], 24)

    def test_valid_reviewed_row(self):
        self.assertEqual(audit_rows([valid_row()])["errors"], [])

    def test_generic_url_blocked(self):
        row = valid_row()
        row["source_url"] = "https://dailymed.nlm.nih.gov/"
        self.assertTrue(audit_rows([row])["errors"])

    def test_missing_source_blocked(self):
        row = valid_row()
        row["source_url"] = ""
        self.assertTrue(audit_rows([row])["errors"])

    def test_unreviewed_claim_blocked(self):
        row = valid_row()
        row["status"] = "PENDING_REVIEW"
        self.assertTrue(audit_rows([row])["errors"])

    def test_reviewed_insufficient_blocked(self):
        row = valid_row()
        row["recommendation"] = "INSUFFICIENT_DATA"
        self.assertTrue(audit_rows([row])["errors"])

    def test_pending_cannot_have_trimester_claim(self):
        row = valid_row()
        row["status"] = "PENDING_REVIEW"
        row["recommendation"] = "INSUFFICIENT_DATA"
        row["evidence_level"] = "UNKNOWN"
        row["trimester_1"] = "No evidence"
        self.assertTrue(audit_rows([row])["errors"])

    def test_duplicate_ids_blocked(self):
        self.assertTrue(audit_rows([valid_row(), valid_row()])["errors"])

    def test_identity_mismatch_blocked(self):
        row = valid_row()
        row["source_authority"] = "NO MATCH EXACTO"
        self.assertTrue(audit_rows([row])["errors"])

    def test_https_required(self):
        row = valid_row()
        row["source_url"] = "http://example.org/label"
        self.assertTrue(audit_rows([row])["errors"])

    def test_generic_or_invalid(self):
        self.assertTrue(generic_or_invalid_url("https://dailymed.nlm.nih.gov/"))
        self.assertTrue(generic_or_invalid_url("https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm"))
        self.assertFalse(generic_or_invalid_url("https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=drug"))


if __name__ == "__main__":
    unittest.main()
