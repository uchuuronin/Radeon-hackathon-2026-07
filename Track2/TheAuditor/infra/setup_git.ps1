# TheAuditor — Day 1 git setup (PowerShell)
#
# Run from:  D:\Celestia\Projects\AMDHackathon\Radeon-hackathon-2026-07
#            (the REPO ROOT, not Track2\TheAuditor)
#
# Read it before running it. It stops at the first failed gate rather than
# pushing something wrong.

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# GATE 1 — are we in a clone of OUR FORK, or of upstream?
# ---------------------------------------------------------------------------
# If 'origin' points at AMD-DEV-CONTEST you cannot push, and you will find
# that out after doing an hour of work. Check now.

Write-Host "`n=== Remotes ===" -ForegroundColor Cyan
git remote -v

Write-Host @"

Expected:
  origin    https://github.com/<YOUR-USERNAME>/Radeon-hackathon-2026-07.git
  upstream  https://github.com/AMD-DEV-CONTEST/Radeon-hackathon-2026-07.git

If origin is AMD-DEV-CONTEST, fix it before continuing:
  git remote rename origin upstream
  git remote add origin https://github.com/<YOUR-USERNAME>/Radeon-hackathon-2026-07.git

If upstream is missing:
  git remote add upstream https://github.com/AMD-DEV-CONTEST/Radeon-hackathon-2026-07.git

"@ -ForegroundColor Yellow

Read-Host "Press Enter once remotes are correct (Ctrl+C to abort)"

# ---------------------------------------------------------------------------
# GATE 2 — identity. Commits must be attributable to a registered participant.
# ---------------------------------------------------------------------------
Write-Host "`n=== Identity ===" -ForegroundColor Cyan
git config user.name
git config user.email
Write-Host @"

This email should match the GitHub account you registered with AMD.
To set it for this repo only:
  git config user.name  "Your Name"
  git config user.email "you@example.com"

"@ -ForegroundColor Yellow

Read-Host "Press Enter to continue"

# ---------------------------------------------------------------------------
# Work on a branch, never on main.
# ---------------------------------------------------------------------------
git fetch upstream
git checkout -B track2-theauditor upstream/main

# ---------------------------------------------------------------------------
# Scaffold — matches the A/B seam. Directories are physical, not conventional.
# ---------------------------------------------------------------------------
$root = "Track2\TheAuditor"

$dirs = @(
  "$root\src\theauditor",
  "$root\src\theauditor\verify",      # A
  "$root\src\theauditor\linker",      # A
  "$root\src\theauditor\reconcile",   # A
  "$root\src\theauditor\extract",     # B — only place that imports a model client
  "$root\src\theauditor\console",     # A (S3)
  "$root\data\generator",             # A
  "$root\data\fixtures\records",      # A4 — Day 1 deliverable
  "$root\data\fixtures\corrupted",    # A6
  "$root\data\fixtures\holdout",      # Layout C, sealed
  "$root\tests",
  "$root\bench",                      # B
  "$root\infra",                      # B — versions.md
  "$root\docs"
)
foreach ($d in $dirs) { New-Item -ItemType Directory -Force -Path $d | Out-Null }

# Git does not track empty directories; .gitkeep makes the structure land in
# the first commit so B sees the layout immediately.
foreach ($d in $dirs) {
  $keep = Join-Path $d ".gitkeep"
  if (-not (Test-Path (Join-Path $d "*"))) { New-Item -ItemType File -Force -Path $keep | Out-Null }
}

# Python packages need __init__.py
@(
  "$root\src\theauditor\__init__.py",
  "$root\src\theauditor\verify\__init__.py",
  "$root\src\theauditor\linker\__init__.py",
  "$root\src\theauditor\reconcile\__init__.py",
  "$root\src\theauditor\extract\__init__.py",
  "$root\src\theauditor\console\__init__.py"
) | ForEach-Object { if (-not (Test-Path $_)) { New-Item -ItemType File -Force -Path $_ | Out-Null } }

# The holdout seal
@"
# HOLDOUT — DO NOT OPEN UNTIL DEMO DAY

Layout C lives here. It is reserved from development so the
"what about a layout you didn't design for?" question can be answered
live, on camera, with evidence instead of assertion.

Opening this during development destroys the only unseen-layout
evidence the project has. There is no way to un-see it.
"@ | Set-Content -Encoding UTF8 "$root\data\fixtures\holdout\README.md"

Write-Host "`nScaffold created." -ForegroundColor Green

# ---------------------------------------------------------------------------
# GATE 3 — move the contract into place and PROVE IT PASSES before committing
# ---------------------------------------------------------------------------
Write-Host @"

MANUAL STEP — do this now, before the commit:

  1. Confirm you have schemas.py v1.1 (about 21,900 bytes, contains the
     string SCHEMA_VERSION = "1.1"). The v1.0 file is ~11,980 bytes and
     carries the broken subtotal + tax = total identity.

  2. Move the two files into place:
       Move-Item $root\schemas.py                 $root\src\theauditor\schemas.py -Force
       Move-Item $root\test_schemas_contract.py   $root\tests\test_schemas_contract.py -Force

  3. Run the tests. Do not commit red tests.
       cd $root
       python -m pytest tests -q

"@ -ForegroundColor Yellow

Read-Host "Press Enter once tests are GREEN (Ctrl+C to abort)"

# ---------------------------------------------------------------------------
# GATE 4 — what is actually about to be committed?
# ---------------------------------------------------------------------------
git add -A -- "$root"

Write-Host "`n=== Staged files ===" -ForegroundColor Cyan
git diff --cached --name-only

Write-Host "`n=== Anything staged OUTSIDE Track2/TheAuditor? (must be empty) ===" -ForegroundColor Cyan
git diff --cached --name-only | Where-Object { $_ -notlike "Track2/TheAuditor/*" }

Write-Host "`n=== Files over 5 MB (must be empty) ===" -ForegroundColor Cyan
git diff --cached --name-only | ForEach-Object {
  if (Test-Path $_) {
    $f = Get-Item $_
    if ($f.Length -gt 5MB) { "{0}  {1:N1} MB" -f $f.Name, ($f.Length / 1MB) }
  }
}

Write-Host @"

Review the three lists above.
  - Anything outside Track2/TheAuditor/ -> unstage it. The PR must not
    touch shared files; a maintainer merging 13 PRs will reject conflicts.
  - Anything large -> it is a model weight or a dataset. Remove it and add
    it to .gitignore.

"@ -ForegroundColor Yellow

Read-Host "Press Enter to commit"

git commit -m "Track2/TheAuditor: freeze schemas.py v1.1 contract and project scaffold

The single shared interface between the data/logic half and the GPU half.
CanonicalDoc, LineItem, answer-key shapes, verification report, and the
tolerance model, with a contract test suite that runs with no GPU and no
network.

Field semantics follow EN 16931 business terms; the verifier's arithmetic
identities implement BR-CO-10, BR-CO-13 and BR-CO-15. Document types follow
UBL 2.1 naming across the quote-to-cash chain.

Frozen at SCHEMA_VERSION 1.1. Changes require both sign-offs and a version
bump, because the schema text is embedded in the extraction prompt prefix
and any edit invalidates prefix-cache hits and prior benchmarks."

git push -u origin track2-theauditor

Write-Host @"

Pushed. Now open the DRAFT pull request:

  https://github.com/AMD-DEV-CONTEST/Radeon-hackathon-2026-07/compare

  base: AMD-DEV-CONTEST/Radeon-hackathon-2026-07  main
  head: <YOUR-USERNAME>/Radeon-hackathon-2026-07  track2-theauditor

  Title: Track 2, <TEAM NAME>, The Auditor
  Body:  paste from PR_TEMPLATE.md
  Use the dropdown next to the green button -> "Create draft pull request"

"@ -ForegroundColor Green
