from rsipo.placement import node_resource, ray_nodes


def test_unpinned_requests_nothing(monkeypatch):
    monkeypatch.delenv("RSIPO_RAY_NODES", raising=False)
    assert ray_nodes() == []
    assert node_resource(5) == {}


def test_empty_value_is_unpinned(monkeypatch):
    monkeypatch.setenv("RSIPO_RAY_NODES", "")
    assert node_resource(0) == {}


def test_pinned_requests_cycle_over_nodes(monkeypatch):
    monkeypatch.setenv("RSIPO_RAY_NODES", "10.0.0.1, 10.0.0.2")
    assert ray_nodes() == ["10.0.0.1", "10.0.0.2"]
    assert [node_resource(i) for i in range(3)] == [{"node:10.0.0.1": 0.001}, {"node:10.0.0.2": 0.001}, {"node:10.0.0.1": 0.001}]
