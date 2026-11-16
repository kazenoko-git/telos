import pytest
from scripts.release_notes import release_notes


def test_extracts_only_selected_version():
    changelog = '## [1.5.1] - 2026-11-13\n\n### Fixed\n- Resume.\n\n## [1.5.0]\n\n- Sandbox.\n\n[Unreleased]: https://example.com\n'
    assert release_notes(changelog, 'v1.5.1') == '### Fixed\n- Resume.\n'
    assert release_notes(changelog, 'v1.5.0') == '- Sandbox.\n'
    with pytest.raises(ValueError):
        release_notes(changelog, 'v1.6.0')
    with pytest.raises(ValueError):
        release_notes(changelog, 'main')
