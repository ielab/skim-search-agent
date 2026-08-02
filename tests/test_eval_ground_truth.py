from agent_search.corpus.units import CodeUnit
from evaluation.ground_truth import changed_line_ranges, gold_files, gold_units

DIFF = '''diff --git a/pkg/mod.py b/pkg/mod.py
index 1111111..2222222 100644
--- a/pkg/mod.py
+++ b/pkg/mod.py
@@ -5,3 +5,4 @@ def foo():
     a = 1
-    b = 2
+    b = 3
+    c = 4
     return a
'''

ADD_ONLY = '''diff --git a/x.py b/x.py
--- a/x.py
+++ b/x.py
@@ -10,0 +11,2 @@ def g():
+    new = 1
+    line = 2
'''


def test_changed_ranges_are_edited_lines_only():
    # hunk spans old lines 5-7 but only line 6 is edited ('-    b = 2'); the
    # context lines 5 and 7 must NOT be gold (LocAgent/Agentless convention)
    assert changed_line_ranges(DIFF)["pkg/mod.py"] == [(6, 6)]


def test_pure_addition_is_zero_width_at_insertion_point():
    # `@@ -10,0 +11,2 @@`: git's old_start for a pure insertion is the line
    # BEFORE the insertion point — the anchor is old line 10
    assert changed_line_ranges(ADD_ONLY)["x.py"] == [(10, 10)]


def test_gold_files():
    assert gold_files(DIFF) == {"pkg/mod.py"}


def test_gold_units_overlap_only():
    units = [
        CodeUnit("pkg/mod.py::foo", "pkg/mod.py", "foo", 4, 9, "..."),
        CodeUnit("pkg/mod.py::bar", "pkg/mod.py", "bar", 20, 30, "..."),
    ]
    g = gold_units(changed_line_ranges(DIFF), {"pkg/mod.py": units})
    assert g == {"pkg/mod.py::foo"}
