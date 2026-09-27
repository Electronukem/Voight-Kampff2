"""Tests for the legibility gate.

The gate decides which scene-cued numbers are allowed to be published, so its
failure modes are asymmetric. Withholding a good number costs a table row.
Publishing a confounded one puts a legibility artefact on a page headed with
refusal rates, where nothing downstream can tell it from a safety finding --
so the cases that matter most here are the ones where evidence is missing,
partial, or contradictory, and the gate has to decide anyway.
"""

from __future__ import annotations

import json

import pytest

from voight_kampff.admissibility import (
    Probe,
    ProbeSet,
    Verdict,
    load_probes,
    probe_document,
    rule_all,
)


def probe(policy: str, resolution: int, ci_low: float, chance: float = 1 / 3) -> Probe:
    return Probe(
        policy=policy,
        resolution=resolution,
        chance=chance,
        point=ci_low + 0.1,
        ci_low=ci_low,
        ci_high=min(1.0, ci_low + 0.2),
        admissible=ci_low > chance,
    )


PASS_224 = probe("p", 224, 0.62)
FAIL_224 = probe("p", 224, 0.20)
PASS_448 = probe("p", 448, 0.90)
FAIL_448 = probe("p", 448, 0.18)


class TestReadingProbeRecords:
    def test_reads_a_probe_result_record(self):
        raw = {
            "policy": "reader",
            "resolution": 336,
            "chance": 0.3333,
            "accuracy": {"point": 0.9, "ci_low": 0.8, "ci_high": 0.96},
            "admissible": True,
        }
        p = Probe.from_dict(raw)
        assert (p.policy, p.resolution, p.admissible) == ("reader", 336, True)
        assert p.ci_low == 0.8

    def test_missing_admissible_flag_is_recomputed_not_defaulted(self):
        """A record without the flag must never read as a pass."""
        below = Probe.from_dict(
            {
                "policy": "p",
                "resolution": 224,
                "chance": 0.5,
                "accuracy": {"point": 0.52, "ci_low": 0.40, "ci_high": 0.64},
            }
        )
        assert below.admissible is False

        above = Probe.from_dict(
            {
                "policy": "p",
                "resolution": 224,
                "chance": 0.5,
                "accuracy": {"point": 0.80, "ci_low": 0.66, "ci_high": 0.90},
            }
        )
        assert above.admissible is True

    def test_summary_quotes_the_interval_the_verdict_turned_on(self):
        assert "[62%, 82%]" in probe("p", 224, 0.62).summary()


class TestRuling:
    def test_exact_resolution_match_decides(self):
        assert ProbeSet((PASS_224,)).verdict("p", 224) is Verdict.ADMISSIBLE
        assert ProbeSet((FAIL_224,)).verdict("p", 224) is Verdict.INADMISSIBLE

    def test_passing_lower_implies_passing_higher(self):
        """More pixels cannot make text harder to read."""
        assert ProbeSet((PASS_224,)).verdict("p", 336) is Verdict.ADMISSIBLE

    def test_failing_higher_implies_failing_lower(self):
        assert ProbeSet((FAIL_448,)).verdict("p", 336) is Verdict.INADMISSIBLE

    def test_failing_lower_says_nothing_about_higher(self):
        """224px illegible does not mean 448px is: that is the whole probe curve."""
        assert ProbeSet((FAIL_224,)).verdict("p", 448) is Verdict.UNPROBED

    def test_passing_higher_says_nothing_about_lower(self):
        assert ProbeSet((PASS_448,)).verdict("p", 224) is Verdict.UNPROBED

    def test_contradictory_evidence_resolves_against_publishing(self):
        """Passing at 224 and failing at 448 is noise, not a licence."""
        assert ProbeSet((PASS_224, FAIL_448)).verdict("p", 336) is Verdict.INADMISSIBLE

    def test_exact_match_outranks_inference(self):
        assert ProbeSet((PASS_224, FAIL_448)).verdict("p", 224) is Verdict.ADMISSIBLE

    def test_unknown_policy_is_unprobed(self):
        assert ProbeSet((PASS_224,)).verdict("other", 224) is Verdict.UNPROBED

    def test_unrecorded_resolution_is_unprobed_however_many_probes(self):
        """An unrecorded render resolution is itself a missing control."""
        assert ProbeSet((PASS_224, PASS_448)).verdict("p", None) is Verdict.UNPROBED

    def test_ruling_cites_the_probe_it_used(self):
        ruling = ProbeSet((PASS_224,)).rule("p", 336)
        assert ruling.probe is PASS_224
        assert ruling.exact is False
        assert "224px" in ruling.reason()

    def test_unprobed_ruling_explains_the_absence(self):
        assert "no legibility probe" in ProbeSet().rule("p", 336).reason()

    def test_only_admissible_is_publishable(self):
        assert Verdict.ADMISSIBLE.is_publishable
        assert not Verdict.INADMISSIBLE.is_publishable
        assert not Verdict.UNPROBED.is_publishable


class TestRuleAll:
    def test_no_probe_file_still_rules_on_every_policy(self):
        rulings = rule_all(None, ["a", "b"], 336)
        assert [r.verdict for r in rulings.values()] == [Verdict.UNPROBED] * 2

    def test_rules_each_policy_independently(self):
        probes = ProbeSet((PASS_224, probe("q", 224, 0.20)))
        rulings = rule_all(probes, ["p", "q", "r"], 224)
        assert rulings["p"].verdict is Verdict.ADMISSIBLE
        assert rulings["q"].verdict is Verdict.INADMISSIBLE
        assert rulings["r"].verdict is Verdict.UNPROBED


class TestLoading:
    @pytest.fixture
    def record(self) -> dict:
        return {
            "policy": "reader",
            "resolution": 336,
            "chance": 1 / 3,
            "accuracy": {"point": 0.9, "ci_low": 0.8, "ci_high": 0.96},
            "admissible": True,
        }

    def test_reads_the_envelope_vk_probe_writes(self, tmp_path, record):
        path = tmp_path / "probe.json"
        path.write_text(json.dumps(probe_document([record], choices=3)), encoding="utf-8")
        probes = load_probes(path)
        assert probes.policies() == ("reader",)
        assert probes.verdict("reader", 336) is Verdict.ADMISSIBLE

    def test_reads_a_bare_list(self, tmp_path, record):
        """A policy probed elsewhere should not need an envelope to be usable."""
        path = tmp_path / "probe.json"
        path.write_text(json.dumps([record]), encoding="utf-8")
        assert load_probes(path).verdict("reader", 336) is Verdict.ADMISSIBLE

    def test_probe_document_keeps_metadata_beside_results(self, record):
        doc = probe_document([record], policy="template-reader", choices=3)
        assert doc["choices"] == 3 and doc["results"] == [record]

    def test_empty_probe_set_is_falsy(self):
        assert not ProbeSet()
        assert ProbeSet((PASS_224,))
