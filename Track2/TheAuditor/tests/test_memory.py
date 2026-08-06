"""Mechanism C: gated, signature-keyed memory. No GPU, no network.

Most of these assert that memory REFUSES. Unfiltered agent memory is a
documented poisoning vector, and the only thing separating curated precedent
lookup from that is the set of writes and reads this store declines.
"""
from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harmonize.memory import (MemoryPolicy, SignatureMemory,  # noqa: E402
                              resolve_from_memory, signature_of)
from schemas import (AnomalyType, ChainVerdict, Discrepancy,  # noqa: E402
                     ResolutionKind)

PARTY = "Averill Fastener GmbH"


def _d(kind=AnomalyType.PRICE_DRIFT, path="total"):
    return Discrepancy(anomaly_type=kind, doc_ids_involved=["D-3", "D-5"],
                       field_path=path, delta=Decimal("42.00"))


def _fill(m, d, people, kind=ResolutionKind.WITHIN_AGREED_TERMS, party=PARTY):
    for who in people:
        m.approve(d, party, kind, who, pattern="freight billed separately")
    return m


class TestTheGate:

    def test_an_unnamed_approver_is_refused(self):
        """"Approved by someone" is not an audit trail, and every downstream
        auto-resolution cites this name."""
        with pytest.raises(ValueError):
            SignatureMemory().approve(_d(), PARTY,
                                      ResolutionKind.ACCEPT_AS_BILLED, "   ")

    def test_there_is_no_autonomous_write_path(self):
        """`approve` is the only way in, by construction. If a caller could add
        an episode without a human, provenance becomes decoration."""
        assert [n for n in dir(SignatureMemory)
                if not n.startswith("_") and "approve" in n] == ["approve"]


class TestMigrationIsCountableNotAJudgement:

    def test_below_k_is_real_knowledge_and_still_not_enough(self):
        """Two approvals is not nothing, and it is also not settled. Returning
        it would let a caller mistake "seen twice" for "safe to skip the
        model"."""
        m = _fill(SignatureMemory(), _d(), ["chiya", "sam"])
        assert m.lookup(_d(), PARTY) is None

    def test_k_approvals_from_enough_people_migrates(self):
        m = _fill(SignatureMemory(), _d(), ["chiya", "sam", "ada"])
        p = m.lookup(_d(), PARTY)
        assert p is not None and p.is_migrated()

    def test_one_person_confirming_themselves_k_times_does_not_migrate(self):
        """Repetition is not corroboration, and it is the cheapest poisoning
        path available."""
        m = _fill(SignatureMemory(), _d(), ["chiya", "chiya", "chiya"])
        assert m.lookup(_d(), PARTY) is None

    def test_escalation_never_migrates_however_often_it_is_recorded(self):
        """ESCALATE_TO_VENDOR is in the enum so it can be RECORDED without ever
        counting: a pattern nobody has settled must not become an auto-resolve.
        """
        m = _fill(SignatureMemory(), _d(), ["a", "b", "c", "d"],
                  kind=ResolutionKind.ESCALATE_TO_VENDOR)
        assert m.lookup(_d(), PARTY) is None


class TestLookupIsExactAndRefusesAmbiguity:

    def test_a_different_anomaly_type_does_not_match(self):
        m = _fill(SignatureMemory(), _d(), ["a", "b", "c"])
        assert m.lookup(_d(kind=AnomalyType.NEAR_DUPLICATE), PARTY) is None

    def test_a_different_party_does_not_match(self):
        m = _fill(SignatureMemory(), _d(), ["a", "b", "c"])
        assert m.lookup(_d(), "Totally Different Ltd") is None

    def test_a_legal_form_variation_reaches_the_same_precedent(self):
        """One vendor written two ways in two systems is the commonest way a
        precedent gets split in half and never migrates."""
        m = _fill(SignatureMemory(), _d(), ["a", "b", "c"])
        assert m.lookup(_d(), "Averill Fastener") is not None

    def test_two_migrated_resolutions_for_one_case_refuse_rather_than_pick(self):
        """Disagreeing precedent is not precedent. Choosing the more common one
        would let a majority quietly overwrite a minority finding that was
        right."""
        m = SignatureMemory()
        _fill(m, _d(), ["a", "b", "c"], ResolutionKind.WITHIN_AGREED_TERMS)
        _fill(m, _d(), ["d", "e", "f"], ResolutionKind.REQUEST_CREDIT)
        assert m.lookup(_d(), PARTY) is None


class TestTheAuditLine:

    def test_it_names_the_people_and_quotes_how_they_said_it(self):
        """A memory that cannot produce this sentence has no business skipping
        the model."""
        line = _fill(SignatureMemory(), _d(),
                     ["chiya", "sam", "ada"]).lookup(_d(), PARTY).audit_line()
        assert "chiya" in line and "sam" in line and "3 prior approval" in line
        assert "freight billed separately" in line

    def test_the_signature_carries_values_not_python_reprs(self):
        """It is the migration key AND appears in audit lines humans read."""
        sig = signature_of(PARTY, AnomalyType.PRICE_DRIFT,
                           ResolutionKind.WITHIN_AGREED_TERMS)
        assert sig == "averill fastener gmbh|price_drift|within_agreed_terms"
        assert "AnomalyType" not in sig


class TestPersistenceAndTheLadderRung:

    def test_the_store_round_trips_through_a_local_file(self, tmp_path):
        """Local JSONL, no external service. A store that phoned out to embed
        would silently break the whole no-egress claim."""
        m = _fill(SignatureMemory(), _d(), ["a", "b", "c"])
        n = m.save(tmp_path / "mem.jsonl")
        back = SignatureMemory.load(tmp_path / "mem.jsonl")
        assert n == 3 and back.lookup(_d(), PARTY) is not None

    def test_the_rung_splits_settled_from_still_needs_work(self):
        """This is what makes cost per document fall as the store fills:
        extraction is flat, only the ladder is variable."""
        settled, open_ = _d(), _d(kind=AnomalyType.TERM_CHANGE,
                                  path="payment_terms")
        m = _fill(SignatureMemory(), settled, ["a", "b", "c"])
        v = ChainVerdict(chain_id="CH-1", doc_ids=["D-3", "D-5"],
                         discrepancies=[settled, open_])
        remaining, resolved = resolve_from_memory(v, PARTY, m)
        assert [d.field_path for d in remaining] == ["payment_terms"]
        assert len(resolved) == 1 and "prior approval" in resolved[0]

    def test_an_empty_store_resolves_nothing(self):
        v = ChainVerdict(chain_id="CH-1", doc_ids=["D-3", "D-5"],
                         discrepancies=[_d()])
        remaining, resolved = resolve_from_memory(v, PARTY, SignatureMemory())
        assert len(remaining) == 1 and not resolved
