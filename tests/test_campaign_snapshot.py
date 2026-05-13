from pathlib import Path

from propab import config
from propab.campaign import ResearchCampaign
from propab.campaign_snapshot import read_snapshot, write_campaign_snapshot
from propab.hypothesis_tree import HypothesisNode, HypothesisTree


def test_write_read_snapshot_roundtrip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config.settings, "propab_data_dir", str(tmp_path))
    tree = HypothesisTree()
    tree.nodes["c1"] = HypothesisNode(id="c1", text="x", parent_id=None, depth=0, verdict="confirmed")
    tree.confirmed.append("c1")
    c = ResearchCampaign(id="00000000-0000-0000-0000-0000000000ab", question="Q", hypothesis_tree=tree)
    p = write_campaign_snapshot("unit", c, {"k": []})
    assert p is not None
    raw2, c2, pr = read_snapshot(p)
    assert raw2["phase"] == "unit"
    assert c2.question == "Q"
    assert "c1" in c2.hypothesis_tree.confirmed
    assert pr == {"k": []}
