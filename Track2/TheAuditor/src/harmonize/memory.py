"""Signature-keyed memory of approved resolutions. Mechanism C. No GPU.

WHAT THIS IS, AND WHAT IT IS DELIBERATELY NOT
---------------------------------------------
It is curated precedent lookup. It is NOT "the agent learns from experience",
and the difference is not modesty, it is the security claim.

Unfiltered agent memory is a documented poisoning vector (OWASP ASI06). Small
poisoned sets reliably outcompete benign experiences in retrieval, trigger-
activated demonstrations can be implanted, and benign accumulation alone
dilutes safety cues over time with no attacker involved. There is also little
peer-reviewed evidence that naive episodic memory improves accuracy without
heavy curation. So every entry here requires a named human approval, carries
its provenance, and is retrieved by EXACT signature rather than by embedding
similarity hoping to generalise. The gate is what turns the vulnerability into
an auditable control: every auto-resolution traces to named approvals.

THE SIGNATURE
-------------
    party x anomaly_type x resolution_kind

Canonical, order-independent, and hashed only over things that COLLIDE. The
resolution KIND is an enum for exactly this reason: free text does not collide,
so two analysts describing the same decision differently would never accumulate
the k confirmations migration needs, and the mechanism would silently never
fire. The free-text pattern rides alongside as provenance and is shown, never
hashed.

Party is normalised through the verifier's own comparison, so "Averill Fastener"
and "Averill Fastener GmbH" land on one signature. A vendor written two ways in
two systems is the single most common way a legal-form variation splits what
should be one precedent.

MIGRATION IS COUNTABLE, NOT A JUDGEMENT
---------------------------------------
A signature moves slow -> fast at >= k agreeing settled approvals from at least
`min_approvers` distinct people. Both halves matter. Counting approvals alone
lets one person confirm their own precedent k times, which is not corroboration,
it is repetition. ESCALATE_TO_VENDOR never counts: a pattern nobody has actually
settled must not become an auto-resolve.

WHERE IT LIVES
--------------
`harmonize/`, with reconciliation, because the signature is keyed on
reconciliation output. It is written by the console (on an explicit approval
turn) and read by the ladder (at the memory rung, before any GPU token is
spent).
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

from normalise import compare_parties, normalise_party
from schemas import (AnomalyType, ChainVerdict, Discrepancy, MemoryEpisode,
                     ResolutionKind)


def _v(x) -> str:
    """The enum's VALUE, whichever form it arrives in.

    The contract sets `use_enum_values`, so a field read off a model is already
    a plain string while a freshly constructed enum is not. `str()` on the
    latter yields "AnomalyType.PRICE_DRIFT" under Python 3.12, which would put
    a Python repr in the migration key and in every audit line a human reads.
    """
    return getattr(x, "value", x)


def signature_of(party_name: str, anomaly_type: AnomalyType,
                 resolution_kind: ResolutionKind) -> str:
    """The canonical key. Stable across runs, machines and orderings.

    Deliberately readable rather than a hash digest: this string appears in
    audit lines an analyst reads, and "sig:a3f9e21b" tells them nothing about
    why a case auto-resolved.
    """
    party = normalise_party(party_name or "").replace("|", " ").strip()
    return f"{party}|{_v(anomaly_type)}|{_v(resolution_kind)}"


@dataclass(frozen=True)
class MemoryPolicy:
    """The two numbers that decide when a precedent becomes an auto-resolve."""
    #: Agreeing settled approvals before a signature migrates slow -> fast.
    k: int = 3
    #: Distinct approvers among them. Counting approvals alone lets one person
    #: confirm their own precedent k times, which is repetition rather than
    #: corroboration, and it is the cheapest possible poisoning path.
    min_approvers: int = 2
    #: Fold legal-form variations onto one signature ("Ltd", "GmbH").
    fold_legal_forms: bool = True


DEFAULT_POLICY = MemoryPolicy()


@dataclass
class Precedent:
    """What memory knows about one signature."""
    signature: str
    episodes: list[MemoryEpisode]

    @property
    def settled(self) -> list[MemoryEpisode]:
        return [e for e in self.episodes
                if ResolutionKind(e.resolution_kind).settles]

    @property
    def approvers(self) -> set[str]:
        return {e.approver for e in self.settled}

    def is_migrated(self, policy: MemoryPolicy = DEFAULT_POLICY) -> bool:
        return (len(self.settled) >= policy.k
                and len(self.approvers) >= policy.min_approvers)

    def audit_line(self) -> str:
        """The sentence that justifies an auto-resolution to a human.

        Names the people, quotes how they described it, and states the count.
        A memory that cannot produce this sentence has no business skipping the
        model.
        """
        names = ", ".join(sorted(self.approvers))
        patterns = [e.pattern for e in self.settled if e.pattern]
        how = f", recorded as {patterns[0]!r}" if patterns else ""
        return (f"auto-resolved: signature {self.signature!r} has "
                f"{len(self.settled)} prior approval(s) by [{names}]{how}")


class SignatureMemory:
    """Gated, append-only store of approved resolutions.

    Append-only on purpose. A precedent that can be edited in place cannot be
    audited: the record of what was believed when a case was auto-resolved must
    survive somebody changing their mind afterwards. Withdrawal is a new
    episode, not a deletion.
    """

    def __init__(self, policy: MemoryPolicy = DEFAULT_POLICY) -> None:
        self.policy = policy
        self._by_signature: dict[str, list[MemoryEpisode]] = defaultdict(list)

    # --- writing -----------------------------------------------------------

    def approve(self, discrepancy: Discrepancy, party_name: str,
                resolution_kind: ResolutionKind, approver: str,
                pattern: str = "", chain_id: str = "") -> MemoryEpisode:
        """Record ONE human-approved resolution. The only way in.

        There is no autonomous write path and there is not meant to be. If a
        caller could add an episode without an approver, the provenance in
        every audit line downstream would be a decoration rather than a fact.
        """
        if not approver or not approver.strip():
            raise ValueError(
                "an episode needs a named approver: gated entry is the control "
                "that makes auto-resolution defensible, and 'approved by "
                "someone' is not an audit trail")
        sig = signature_of(party_name, AnomalyType(discrepancy.anomaly_type),
                           resolution_kind)
        ep = MemoryEpisode(
            signature=sig, party_name=party_name,
            anomaly_type=AnomalyType(discrepancy.anomaly_type),
            resolution_kind=resolution_kind, pattern=pattern.strip(),
            approver=approver.strip(), chain_id=chain_id,
            field_path=discrepancy.field_path)
        self._by_signature[sig].append(ep)
        return ep

    # --- reading -----------------------------------------------------------

    def _candidate_signatures(self, party_name: str,
                              anomaly_type: AnomalyType) -> list[str]:
        """Signatures for this party and anomaly, across resolution kinds.

        Legal-form folding happens here rather than in `signature_of`, so the
        stored key stays exactly what was approved while lookup can still reach
        it from a differently-written name.
        """
        want = normalise_party(party_name or "")
        out = []
        for sig in self._by_signature:
            party, atype, _kind = sig.split("|", 2)
            if atype != _v(anomaly_type):
                continue
            if party == want:
                out.append(sig)
            elif self.policy.fold_legal_forms and party and want:
                m = compare_parties(party, want)
                if m.exact or m.same_base:
                    out.append(sig)
        return sorted(out)

    def lookup(self, discrepancy: Discrepancy,
               party_name: str) -> Optional[Precedent]:
        """EXACT signature lookup, and nothing fuzzier.

        Returns the migrated precedent only. A signature with two approvals is
        real knowledge and still not enough to skip the model, so it is not
        returned: the caller must not be able to mistake "we have seen this
        twice" for "this is settled".

        When several resolution kinds have migrated for one party and anomaly,
        the store refuses rather than picking. Disagreeing precedent is not a
        precedent, and choosing the more common one would let a majority quietly
        overwrite a minority finding that was correct.
        """
        migrated = [Precedent(s, list(self._by_signature[s]))
                    for s in self._candidate_signatures(
                        party_name, AnomalyType(discrepancy.anomaly_type))]
        migrated = [p for p in migrated if p.is_migrated(self.policy)]
        if len(migrated) != 1:
            return None
        return migrated[0]

    def precedent_for_signature(self, signature: str) -> Optional[Precedent]:
        if signature not in self._by_signature:
            return None
        return Precedent(signature, list(self._by_signature[signature]))

    # --- persistence -------------------------------------------------------

    def save(self, path: Path) -> int:
        """JSONL, local file, no external service.

        A memory store that phoned out to embed would silently break the entire
        no-egress claim, which is why retrieval here is exact-match over local
        state and there is no embedding step at all.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        eps = [e for v in self._by_signature.values() for e in v]
        path.write_text("\n".join(e.model_dump_json() for e in eps) + "\n",
                        encoding="utf-8")
        return len(eps)

    @classmethod
    def load(cls, path: Path,
             policy: MemoryPolicy = DEFAULT_POLICY) -> "SignatureMemory":
        m = cls(policy)
        if not path.exists():
            return m
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                ep = MemoryEpisode.model_validate_json(line)
                m._by_signature[ep.signature].append(ep)
        return m

    # --- reporting ---------------------------------------------------------

    @property
    def signatures(self) -> list[str]:
        return sorted(self._by_signature)

    def migrated_signatures(self) -> list[str]:
        return [s for s in self.signatures
                if Precedent(s, self._by_signature[s]).is_migrated(self.policy)]

    def render(self) -> str:
        L = [f"  signatures held      : {len(self.signatures)}",
             f"  migrated (fast path) : {len(self.migrated_signatures())} "
             f"(>= {self.policy.k} settled approvals from >= "
             f"{self.policy.min_approvers} people)"]
        for s in self.migrated_signatures()[:10]:
            L.append(f"    {Precedent(s, self._by_signature[s]).audit_line()}")
        return "\n".join(L)


def resolve_from_memory(verdict: ChainVerdict, party_name: str,
                        memory: SignatureMemory
                        ) -> tuple[list[Discrepancy], list[str]]:
    """Split a verdict into (still needs work, audit lines for what memory
    settled).

    This is the memory RUNG of the escalation ladder, and it sits above the
    free deterministic checks and below any GPU call. A discrepancy matched by
    a migrated precedent costs a table lookup; everything else falls through to
    inference. That, and only that, is what makes cost per document fall as the
    store fills, because extraction is flat and only the ladder is variable.
    """
    remaining: list[Discrepancy] = []
    resolved: list[str] = []
    for d in verdict.discrepancies:
        p = memory.lookup(d, party_name)
        if p is None:
            remaining.append(d)
            continue
        resolved.append(f"{d.field_path}: {p.audit_line()}")
    return remaining, resolved
