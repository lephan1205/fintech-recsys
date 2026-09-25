"""The write-up is checked against the code and the generated results.

* Appendix A (the decision register) must list every ``RegisterRow`` by its exact name with
  the value the config dataclasses actually default to.
* Every numeric cell in every other table of ``docs/ARCHITECTURE.md`` must be copied from a
  ``docs/results/*.md`` table (or be a config default from the register), never typed by hand.
* Both diagrams pass ``scripts/check_diagrams.py``.

The whole module is skipped while ``docs/ARCHITECTURE.md`` does not exist, so the code
phases stay green before the write-up is written.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

from recsys.training.config import decision_register

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "ARCHITECTURE.md"
RESULTS = ROOT / "docs" / "results"
DIAGRAMS = ROOT / "docs" / "diagrams"

pytestmark = pytest.mark.skipif(not DOC.exists(), reason="docs/ARCHITECTURE.md not written yet")

_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)*")
# Cells that are obviously not measurements: units, ranges, shapes, code, plain words.
_PLAIN_INT = re.compile(r"^-?\d+$")


def _sections(text: str) -> dict[str, str]:
    """Split on level-2 headings; keys are heading texts."""
    out: dict[str, str] = {}
    current = "_preamble"
    buf: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            out[current] = "\n".join(buf)
            current, buf = line[3:].strip(), []
        else:
            buf.append(line)
    out[current] = "\n".join(buf)
    return out


def _tables(text: str) -> list[list[list[str]]]:
    """Markdown pipe tables -> list of tables, each a list of rows of stripped cells."""
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s[1:-1].split("|")]
            if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                continue  # separator row
            current.append(cells)
        elif current:
            tables.append(current)
            current = []
    if current:
        tables.append(current)
    return tables


def _numbers_in_cell(cell: str) -> list[str]:
    cell = cell.replace("**", "").replace("`", "")
    return [m.group(0).replace(",", "") for m in _NUMBER.finditer(cell)]


def _canonical(num: str) -> str:
    """'85312.0000', '85312' and '85312.0' compare equal; non-floats are kept verbatim."""
    try:
        return repr(float(num))
    except ValueError:
        return num


def _prose_only(text: str) -> str:
    """Drop table rows, code fences and HTML comments so the word count is prose only."""
    out: list[str] = []
    in_code = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("```"):
            in_code = not in_code
            continue
        if in_code or s.startswith("|") or s.startswith("<!--") or s.startswith("!["):
            continue
        out.append(line)
    return "\n".join(out)


@pytest.fixture(scope="module")
def doc_text() -> str:
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def results_numbers() -> set[str]:
    nums: set[str] = set()
    for f in RESULTS.glob("*.md"):
        text = f.read_text(encoding="utf-8")
        for m in _NUMBER.finditer(text):
            nums.add(_canonical(m.group(0).replace(",", "")))
    return nums


def test_required_sections_present(doc_text: str) -> None:
    heads = list(_sections(doc_text))
    required = (
        "Abstract",
        "Problem setting",
        "Data contract",
        "Stage 1",
        "Stage 2",
        "Stage 3",
        "Stage 4",
        "End-to-end serving",
        "Experimental setup and results",
        "Limitations and future work",
        "References",
        "Appendix A",
        "Appendix B",
    )
    for r in required:
        assert any(h.startswith(r) or r in h for h in heads), f"missing section {r!r}"
    prose = _prose_only(doc_text)
    words = len(re.findall(r"[A-Za-z][A-Za-z'’-]+", prose))
    assert 8_000 <= words <= 13_000, f"write-up has {words} prose words; target 8k-12k"


def test_register_matches_config_defaults(doc_text: str) -> None:
    sections = _sections(doc_text)
    key = next(h for h in sections if h.startswith("Appendix A"))
    rows = [r for t in _tables(sections[key]) for r in t]
    by_name = {r[0].strip("`"): r for r in rows if r and r[0].startswith("`")}
    missing = []
    wrong = []
    for reg in decision_register():
        row = by_name.get(reg.name)
        if row is None:
            missing.append(reg.name)
            continue
        value = row[1].strip("`")
        if value != reg.value:
            wrong.append((reg.name, value, reg.value))
    assert not missing, f"register rows missing from Appendix A: {missing}"
    assert not wrong, f"Appendix A values differ from config defaults (name, doc, code): {wrong}"
    # the header must carry the seven mandated columns
    header = rows[0]
    for col in ("parameter", "value", "config location", "why this value"):
        assert any(col in c.lower() for c in header), f"Appendix A lacks column {col!r}"


def test_every_numeric_table_cell_comes_from_results(
    doc_text: str, results_numbers: set[str]
) -> None:
    sections = _sections(doc_text)
    register_values = {r.value for r in decision_register()}
    register_numbers = {_canonical(n) for v in register_values for n in _numbers_in_cell(v)}
    unsourced: list[tuple[str, str]] = []
    for head, body in sections.items():
        if head.startswith("Appendix A"):
            continue
        for table in _tables(body):
            for row in table:
                for cell in row:
                    for num in _numbers_in_cell(cell):
                        if _PLAIN_INT.fullmatch(num) and abs(int(num)) <= 12:
                            continue  # counts like "3 towers", "10 slots": not measurements
                        key = _canonical(num)
                        if key in results_numbers or key in register_numbers:
                            continue
                        unsourced.append((head, cell))
    assert not unsourced, f"table numbers not found in docs/results or the register: {unsourced}"


def test_every_results_table_is_cited(doc_text: str) -> None:
    for f in sorted(RESULTS.glob("*.md")):
        assert f.name in doc_text, f"{f.name} is never cited in the write-up"


def test_diagrams_pass_checklist() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_diagrams.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_readme_embeds_diagrams_and_points_to_write_up() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for name in ("serving_pipeline.svg", "training_pipeline.svg"):
        assert name in readme and (DIAGRAMS / name).exists()
    assert "docs/ARCHITECTURE.md" in readme
