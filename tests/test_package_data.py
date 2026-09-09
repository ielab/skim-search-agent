"""The prompt profiles are package DATA and must ship with the package: the loader resolves
them next to its own __file__, and a wheel without them cannot load a single condition."""
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _package_data_patterns() -> list[str]:
    text = (REPO / "pyproject.toml").read_text()
    m = re.search(r'"agent_search\.prompts"\s*=\s*\[([^\]]*)\]', text)
    assert m, "pyproject.toml has no [tool.setuptools.package-data] entry for agent_search.prompts"
    return re.findall(r'"([^"]+)"', m.group(1))


def test_pyproject_declares_prompt_package_data():
    text = (REPO / "pyproject.toml").read_text()
    pd = _package_data_patterns()
    assert "*.yaml" in pd and "tasks/*.md" in pd and "skills/*.md" in pd
    assert re.search(r"include-package-data\s*=\s*true", text)
    assert re.search(r'include\s*=\s*\["agent_search\*"\]', text)
    assert re.search(r'skimsearchagent\s*=\s*"agent_search\.cli:main"', text)


def test_every_prompt_file_the_loader_needs_is_matched_by_package_data():
    from agent_search.prompts import loader
    root = Path(loader.PROMPT_ROOT)
    needed = {root / "tools.yaml", root / "conditions.yaml"}
    needed |= set((root / "tasks").glob("*.md"))
    needed |= set((root / "skills").glob("*.md"))
    needed |= set((root / "vendor").rglob("*.yaml"))
    patterns = _package_data_patterns()
    for f in needed:
        rel = f.relative_to(root)
        assert any(rel.match(p) for p in patterns), f"{rel} is not covered by package-data"


@pytest.mark.slow
def test_built_wheel_contains_the_prompt_files(tmp_path):
    """Build a wheel from a clean `git archive` of the working tree and inspect it."""
    src = tmp_path / "src"
    src.mkdir()
    subprocess.run("git archive HEAD | tar -x -C " + str(src), shell=True, cwd=REPO, check=True)
    # the working tree may be ahead of HEAD: overlay tracked files' current content
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True,
                             check=True).stdout.split(b"\0")
    for t in tracked:
        if not t:
            continue
        p = REPO / t.decode()
        if p.is_file():
            dst = src / t.decode()
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(p.read_bytes())
    out = subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                          "-q", "-w", str(tmp_path / "dist"), str(src)],
                         capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        pytest.skip(f"wheel build unavailable here: {out.stderr[-400:]}")
    whl = next((tmp_path / "dist").glob("*.whl"))
    names = zipfile.ZipFile(whl).namelist()
    assert "agent_search/prompts/conditions.yaml" in names
    assert "agent_search/prompts/tools.yaml" in names
    assert any(re.match(r"agent_search/prompts/tasks/.*\.md$", n) for n in names)
    assert any(re.match(r"agent_search/prompts/skills/.*\.md$", n) for n in names)
    assert not any(n.startswith(("evaluation/", "analysis/", "scripts/")) for n in names)
