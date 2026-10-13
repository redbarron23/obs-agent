from scripts.check_markdown_links import local_link_errors


def test_reports_missing_relative_link(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("[missing](docs/missing.md)\n")
    assert local_link_errors(tmp_path) == ["README.md: missing link target: docs/missing.md"]


def test_ignores_external_and_anchor_links(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text("[site](https://example.com) [section](#section)\n")
    assert local_link_errors(tmp_path) == []
