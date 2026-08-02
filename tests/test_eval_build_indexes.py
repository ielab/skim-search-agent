from evaluation.build_indexes import unique_corpora, shard_by_repo
from evaluation.datasets import Instance


def _inst(iid, repo, commit):
    return Instance(iid, repo, commit, "issue", "patch", files={})


def test_unique_corpora_dedups_by_repo_and_commit():
    insts = [
        _inst("a1", "django/django", "c1"),
        _inst("a2", "django/django", "c1"),   # same corpus as a1
        _inst("a3", "django/django", "c2"),   # different commit -> new corpus
        _inst("b1", "psf/requests", "c1"),
    ]
    corpora = unique_corpora(insts)
    keys = [k for k, _ in corpora]
    assert len(corpora) == 3                  # (django,c1), (django,c2), (requests,c1)
    assert keys[0] == "django__django@c1"


def test_shard_by_repo_keeps_a_repo_together():
    corpora = unique_corpora([
        _inst("d1", "django/django", "c1"),
        _inst("d2", "django/django", "c2"),
        _inst("r1", "psf/requests", "c1"),
        _inst("s1", "sympy/sympy", "c1"),
    ])
    s0 = shard_by_repo(corpora, shard=0, nshards=2)
    s1 = shard_by_repo(corpora, shard=1, nshards=2)
    # every corpus assigned exactly once
    assert len(s0) + len(s1) == len(corpora)
    # a repo's corpora never split across shards
    for shard in (s0, s1):
        repos_here = {inst.repo for _, inst in shard}
        for other in (s0, s1):
            if other is not shard:
                assert repos_here.isdisjoint({inst.repo for _, inst in other})
    # both django corpora land together
    django = [k for k, i in corpora if i.repo == "django/django"]
    assert (all(k in {kk for kk, _ in s0} for k in django)
            or all(k in {kk for kk, _ in s1} for k in django))
