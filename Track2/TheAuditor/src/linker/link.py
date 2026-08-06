"""Group documents into the deal each belongs to. No GPU, no model.

THE CLAIM THIS CODE HAS TO EARN
-------------------------------
Every commercial quote-to-cash tool solves linking by controlling both ends:
integrate the systems so the handoff carries a key. That collapses the moment a
document originates outside your walls, which is exactly the wedge this project
takes. So the links here are INFERRED, and inference has to survive the
identifier being wrong, because identifiers are the hardest field class in
every published extraction comparison (0/O, 1/l) and the one a reconciliation
engine cannot afford to lose: a dropped character does not corrupt a number, it
silently detaches a document from its deal and every downstream comparison is
then made against nothing.

THE LADDER, AGAIN
-----------------
Resolution is ordered cheapest and safest first, and every document exits at
the earliest rung that resolves it:

  1. EXACT_REFERENCE     the document cites a number we hold. Free, certain.
  2. REPAIRED_REFERENCE  it cites a number one character-confusion from one we
                         hold, and only one. Cheap, and it targets the exact
                         failure the extraction scorer measures.
  3. ATTRIBUTE_MATCH     no usable reference. Party AND date AND amount must
                         all agree, and the winner must be unique.
  4. isolated            we say so, rather than guessing.

WHY THE FALLBACK IS DELIBERATELY HARD TO SATISFY
------------------------------------------------
The asymmetry matters more than the hit rate. A MISSED link leaves a document
unattached and visible: it surfaces as a singleton and an analyst looks at it.
A WRONG link drags a foreign document into a deal, and every comparison after
that is against the wrong baseline, so the error does not present as a linking
failure at all. It presents as a discrepancy that is not there, in a chain that
looks complete. That is far more expensive to find and far more damaging to
trust, so the fallback requires agreement on all three attributes and a unique
winner, and isolates whenever it is unsure.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not judge. Two invoices against one purchase order is a correctly
linked chain here, not a duplicate: duplicate and conflict semantics belong to
the reconciler. Keeping grouping free of judgement is what stops a linking bug
from presenting as a false discrepancy, and stops a real discrepancy from being
silently repaired by dropping a document out of the chain.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal
from typing import Iterable, Optional, Sequence

from normalise import compare_parties
from schemas import (Chain, CanonicalDoc, CROSS_DOC_TOLERANCE, LinkEdge,
                     LinkMethod, allowed_delta)

#: Character confusions that actually occur in extracted identifiers. Both
#: directions, because we do not know which side was misread. Deliberately NOT
#: a general edit distance: "PO-1001" and "PO-1002" are one edit apart and are
#: different purchase orders, so a plain Levenshtein-1 rule would merge
#: unrelated deals. Restricting repair to glyph confusions keeps the
#: transformation tied to the failure mode it exists for.
LOOKALIKES: dict[str, tuple[str, ...]] = {
    "0": ("O", "D", "Q"), "O": ("0", "D", "Q"),
    "1": ("l", "I", "7"), "l": ("1", "I"), "I": ("1", "l"),
    "5": ("S",), "S": ("5",),
    "8": ("B",), "B": ("8",),
    "2": ("Z",), "Z": ("2",),
    "6": ("G",), "G": ("6",),
}


@dataclass(frozen=True)
class LinkPolicy:
    """Every threshold the fallback uses, in one place and named.

    A tuned constant buried in a function is a number nobody can defend on
    camera. These are the knobs an analyst would ask about.
    """
    #: How far apart two documents in one deal may be dated. Wide, because a
    #: quote-to-payment lifecycle genuinely runs months, and because this is a
    #: NECESSARY condition rather than a scoring signal: its job is to rule out
    #: the obviously unrelated, not to pick a winner.
    date_window_days: int = 180
    #: Party names must match at least this well. compare_parties returns three
    #: outcomes; we accept exact and legal-form-equivalent, never "differs".
    require_party_match: bool = True
    #: An amount on the candidate document must equal an amount somewhere in
    #: the chain, within the cross-document tolerance band.
    require_amount_match: bool = True
    #: Refuse to attach when more than one chain qualifies. Isolating is the
    #: recoverable outcome; a coin-flip merge is not.
    require_unique_winner: bool = True
    #: Allow single-glyph reference repair at all.
    repair_references: bool = True
    #: Allow the attribute fallback at all. Off with repair_references off
    #: makes the linker purely literal, which is the setting that measures the
    #: exact-reference path ALONE. Without a real switch a headline number
    #: cannot say which mechanism earned it, and narrowing the date window to
    #: zero is not a substitute: a document dated inside the chain's own span
    #: still qualifies.
    attribute_fallback: bool = True


DEFAULT_POLICY = LinkPolicy()


class _Union:
    """Union-find. The reference graph is a DAG with a diamond at the invoice
    (it cites both the purchase order and the goods receipt), so a simple
    parent-walk would visit a document twice and a tree structure would not
    hold it. Connected components are the right abstraction: we want the SET
    of documents that reach each other, and direction is not part of that
    question."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def add(self, x: str) -> None:
        self._parent.setdefault(x, x)

    def find(self, x: str) -> str:
        self.add(x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:            # path compression
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[max(ra, rb)] = min(ra, rb)   # deterministic winner

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for x in self._parent:
            out[self.find(x)].append(x)
        return out


def _repair_candidates(ref: str, known: Sequence[str]) -> list[str]:
    """Known document numbers reachable from `ref` by ONE glyph confusion."""
    variants = set()
    for i, ch in enumerate(ref):
        for sub in LOOKALIKES.get(ch, ()):
            variants.add(ref[:i] + sub + ref[i + 1:])
    return sorted(v for v in variants if v in known)


def _amounts(doc: CanonicalDoc) -> list[Decimal]:
    vals = [doc.total, doc.total_excl_tax, doc.subtotal, doc.amount_due,
            doc.paid_amount]
    return [v for v in vals if v is not None]


def _amount_agrees(doc: CanonicalDoc, members: Sequence[CanonicalDoc]) -> bool:
    """Some stated amount on `doc` matches some stated amount in the chain.

    Occurrence-level, not identity-level: a payment states an amount due, an
    invoice states a total, and in a clean deal they are the same number under
    different names. Requiring a NAMED field to match would fail on exactly the
    documents the fallback exists to rescue.
    """
    mine = _amounts(doc)
    if not mine:
        return False
    theirs = [a for m in members for a in _amounts(m)]
    for a in mine:
        for b in theirs:
            band = allowed_delta(b, CROSS_DOC_TOLERANCE, [str(a), str(b)])
            if abs(a - b) <= band:
                return True
    return False


def _dates_agree(doc: CanonicalDoc, members: Sequence[CanonicalDoc],
                 window_days: int) -> bool:
    if doc.doc_date is None:
        return False
    dates = [m.doc_date for m in members if m.doc_date is not None]
    if not dates:
        return False
    return (min(dates) - timedelta(days=window_days) <= doc.doc_date
            <= max(dates) + timedelta(days=window_days))


def _party_agrees(doc: CanonicalDoc, members: Sequence[CanonicalDoc]) -> bool:
    """Party agreement, using the verifier's own three-outcome comparison.

    `exact` OR `same_base` counts. "Averill Fastener" and "Averill Fastener
    GmbH" are almost always the same vendor written two ways, and the verifier
    already treats that as WITHIN_TOLERANCE rather than a mismatch. Demanding
    `exact` here would make the fallback reject the legal-form variation that
    is the single most common way one vendor's name differs across two systems,
    which is precisely the situation the fallback exists for. Party is one of
    three necessary conditions, not the deciding one, so accepting the softer
    match does not on its own merge anything.
    """
    for m in members:
        if not doc.party_name or not m.party_name:
            continue
        match = compare_parties(doc.party_name, m.party_name)
        if match.exact or match.same_base:
            return True
    return False


def _disambiguate(doc: CanonicalDoc, holders: Sequence[str],
                  by_id: dict[str, CanonicalDoc],
                  policy: LinkPolicy) -> Optional[tuple[str, str]]:
    """Pick one holder of a shared document number, or refuse.

    Party first, because it is the strongest independent signal and the one a
    human would use: two deals sharing a number almost never share a
    counterparty. Date proximity second, and only among the party survivors,
    for the residual case where the same vendor reused a number.

    Returns None whenever the winner is not unique. The asymmetry from the
    module docstring applies with full force here: these are documents we KNOW
    belong to different deals, so a wrong pick does not merely fail to link, it
    actively welds two real chains together.
    """
    cands = [by_id[h] for h in holders if h in by_id]
    if doc.party_name:
        same_party = [c for c in cands
                      if c.party_name
                      and (lambda m: m.exact or m.same_base)(
                          compare_parties(doc.party_name, c.party_name))]
        if len(same_party) == 1:
            return same_party[0].doc_id, "exactly one shares the party name"
        if same_party:
            cands = same_party                    # narrowed, keep going

    if doc.doc_date is not None:
        dated = [c for c in cands if c.doc_date is not None]
        if dated:
            nearest = min(dated, key=lambda c: abs((c.doc_date - doc.doc_date).days))
            gap = abs((nearest.doc_date - doc.doc_date).days)
            rivals = [c for c in dated
                      if c is not nearest
                      and abs((c.doc_date - doc.doc_date).days) <= gap + 30]
            # A clear winner means one candidate is far nearer in time than any
            # other. "Nearest" alone is not enough: two candidates 40 and 45
            # days away is a coin flip wearing a decimal point.
            if not rivals and gap <= policy.date_window_days:
                return nearest.doc_id, (f"one dated {gap}d away, no other "
                                        f"within 30d of that")
    return None


def _chain_id(members: Sequence[CanonicalDoc]) -> str:
    """Stable, content-derived, order-independent.

    Deliberately not a counter: the same corpus linked twice, or linked in a
    different order, or split across two runs, must produce the same chain_id
    or nothing downstream can be joined back to it. The lowest doc_id in the
    group is a total order that no amount of reshuffling changes.
    """
    return "CH-" + min(d.doc_id for d in members)


def link(docs: Iterable[CanonicalDoc],
         policy: LinkPolicy = DEFAULT_POLICY) -> list[Chain]:
    """Group `docs` into chains. Deterministic and order-independent."""
    docs = sorted(docs, key=lambda d: d.doc_id)
    by_id = {d.doc_id: d for d in docs}

    # A document number can legitimately appear twice: the near-duplicate case
    # differs only in its number, but a re-issued document may not. Keep every
    # holder so an ambiguous reference can be recognised as ambiguous rather
    # than silently resolving to whichever we indexed last.
    by_number: dict[str, list[str]] = defaultdict(list)
    for d in docs:
        if d.doc_number:
            by_number[d.doc_number].append(d.doc_id)

    uf = _Union()
    for d in docs:
        uf.add(d.doc_id)

    edges: list[LinkEdge] = []
    dangling: list[tuple[str, str]] = []          # (doc_id, stated reference)
    unresolved: list[CanonicalDoc] = []

    for d in docs:
        resolved_any = False
        for ref in (d.references or []):
            holders = by_number.get(ref, [])
            if len(holders) == 1:
                uf.union(d.doc_id, holders[0])
                edges.append(LinkEdge(from_doc_id=d.doc_id,
                                      to_doc_id=holders[0],
                                      method=LinkMethod.EXACT_REFERENCE,
                                      stated_reference=ref))
                resolved_any = True
                continue
            if len(holders) > 1:
                # Ambiguous. NOT rare and not a generator artefact: document
                # numbers are not globally unique in practice, because real
                # systems run per-vendor or per-year sequences. Joining to all
                # holders would merge unrelated deals; refusing outright costs
                # recall on every collision. Disambiguate on an INDEPENDENT
                # attribute instead, and only when it leaves exactly one
                # candidate.
                picked = _disambiguate(d, holders, by_id, policy)
                if picked is not None:
                    target, why = picked
                    uf.union(d.doc_id, target)
                    edges.append(LinkEdge(
                        from_doc_id=d.doc_id, to_doc_id=target,
                        method=LinkMethod.DISAMBIGUATED_REFERENCE,
                        stated_reference=ref,
                        rationale=f"{len(holders)} documents carry number "
                                  f"{ref!r}; {why}"))
                    resolved_any = True
                    continue
                dangling.append((d.doc_id, ref))
                continue

            if policy.repair_references:
                cands = _repair_candidates(ref, by_number.keys())
                if len(cands) == 1 and len(by_number[cands[0]]) == 1:
                    target = by_number[cands[0]][0]
                    uf.union(d.doc_id, target)
                    edges.append(LinkEdge(
                        from_doc_id=d.doc_id, to_doc_id=target,
                        method=LinkMethod.REPAIRED_REFERENCE,
                        stated_reference=ref,
                        rationale=f"stated {ref!r}; exactly one known document "
                                  f"number is one character-confusion away "
                                  f"({cands[0]!r})"))
                    resolved_any = True
                    continue
                # Two or more repair candidates is the dangerous case: it means
                # a guess would be a coin flip between real documents.
            dangling.append((d.doc_id, ref))

        if not resolved_any and not (d.references or []):
            unresolved.append(d)

    # --- fallback, on documents that produced no edge at all ----------------
    # Run against the components as they now stand, so a document attaches to a
    # formed chain rather than to another loose document. Iterating in doc_id
    # order keeps the result independent of input order.
    for d in unresolved if policy.attribute_fallback else ():
        if len(uf.groups()[uf.find(d.doc_id)]) > 1:
            continue                              # something attached to it
        qualifying: list[str] = []
        for root, member_ids in uf.groups().items():
            if root == uf.find(d.doc_id):
                continue
            members = [by_id[i] for i in member_ids if i in by_id]
            if not members:
                continue
            if policy.require_party_match and not _party_agrees(d, members):
                continue
            if not _dates_agree(d, members, policy.date_window_days):
                continue
            if policy.require_amount_match and not _amount_agrees(d, members):
                continue
            qualifying.append(root)

        if len(qualifying) == 1 or (qualifying and
                                    not policy.require_unique_winner):
            target_root = sorted(qualifying)[0]
            target = sorted(uf.groups()[target_root])[0]
            uf.union(d.doc_id, target)
            edges.append(LinkEdge(
                from_doc_id=d.doc_id, to_doc_id=target,
                method=LinkMethod.ATTRIBUTE_MATCH,
                rationale=f"no resolvable reference; party, date window "
                          f"(+/-{policy.date_window_days}d) and a stated "
                          f"amount all agree with this chain, and no other "
                          f"chain qualified"))
        # More than one qualifying chain: isolate. See the module docstring on
        # why a missed link is cheaper than a wrong one.

    dangling_by_root: dict[str, list[str]] = defaultdict(list)
    for doc_id, ref in dangling:
        dangling_by_root[uf.find(doc_id)].append(ref)

    edges_by_root: dict[str, list[LinkEdge]] = defaultdict(list)
    for e in edges:
        edges_by_root[uf.find(e.from_doc_id)].append(e)

    chains: list[Chain] = []
    for root, member_ids in uf.groups().items():
        members = [by_id[i] for i in sorted(member_ids) if i in by_id]
        if not members:
            continue
        chains.append(Chain(
            chain_id=_chain_id(members),
            doc_ids=[d.doc_id for d in members],
            edges=sorted(edges_by_root[root],
                         key=lambda e: (e.from_doc_id, e.to_doc_id)),
            dangling_references=sorted(set(dangling_by_root[root]))))
    return sorted(chains, key=lambda c: c.chain_id)
