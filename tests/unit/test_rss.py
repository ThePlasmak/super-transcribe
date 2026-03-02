"""Unit tests for lib/rss.py — HTTP calls mocked."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from lib.rss import fetch_rss_episodes

VALID_RSS = b"""<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Test Podcast</title>
    <item>
      <title>Episode 1</title>
      <enclosure url="https://example.com/ep1.mp3" type="audio/mpeg"/>
    </item>
    <item>
      <title>Episode 2</title>
      <enclosure url="https://example.com/ep2.mp3" type="audio/mpeg"/>
    </item>
    <item>
      <title>Episode 3</title>
      <enclosure url="https://example.com/ep3.mp3" type="audio/mpeg"/>
    </item>
  </channel>
</rss>"""

NO_ENCLOSURE_RSS = b"""<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <item><title>Ep 1</title></item>
  </channel>
</rss>"""


def _mock_urlopen(xml_data: bytes) -> MagicMock:
    """Return a MagicMock that acts as a urllib context manager.

    rss.py calls ``urllib.request.urlopen(req, timeout=30)`` where ``req``
    is a ``urllib.request.Request`` object, then uses the result as a context
    manager.  This helper wires up both behaviours.
    """
    mock_resp = MagicMock()
    mock_resp.read.return_value = xml_data
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_resp)
    mock_cm.__exit__ = MagicMock(return_value=False)
    return MagicMock(return_value=mock_cm)


# AIDEV-NOTE: rss.py imports urllib.request *inside* fetch_rss_episodes, so
# patching "urllib.request.urlopen" is the correct target (not "lib.rss.urlopen").

# ── Happy path ─────────────────────────────────────────────────────────────────


class TestFetchRssEpisodes:
    def test_valid_feed_returns_tuples(self):
        with patch("urllib.request.urlopen", _mock_urlopen(VALID_RSS)):
            episodes = fetch_rss_episodes("https://x.com/feed.rss", latest=0, quiet=True)
        assert len(episodes) == 3
        assert all(isinstance(e, tuple) and len(e) == 2 for e in episodes)

    def test_latest_slices_correctly(self):
        with patch("urllib.request.urlopen", _mock_urlopen(VALID_RSS)):
            episodes = fetch_rss_episodes("https://x.com/feed.rss", latest=2, quiet=True)
        assert len(episodes) == 2

    def test_latest_zero_returns_all(self):
        with patch("urllib.request.urlopen", _mock_urlopen(VALID_RSS)):
            episodes = fetch_rss_episodes("https://x.com/feed.rss", latest=0, quiet=True)
        assert len(episodes) == 3

    def test_first_episode_url_and_title(self):
        with patch("urllib.request.urlopen", _mock_urlopen(VALID_RSS)):
            episodes = fetch_rss_episodes("https://x.com/feed.rss", latest=1, quiet=True)
        assert episodes[0][0] == "https://example.com/ep1.mp3"
        assert episodes[0][1] == "Episode 1"

    def test_latest_5_default(self):
        # 3 episodes in feed, default latest=5 → min(5, 3) = 3, returns all 3
        with patch("urllib.request.urlopen", _mock_urlopen(VALID_RSS)):
            episodes = fetch_rss_episodes("https://x.com/feed.rss", quiet=True)
        assert len(episodes) == 3


# ── Error paths ────────────────────────────────────────────────────────────────


class TestFetchRssErrors:
    def test_no_enclosures_exits_3(self):
        with patch("urllib.request.urlopen", _mock_urlopen(NO_ENCLOSURE_RSS)):
            with pytest.raises(SystemExit) as exc:
                fetch_rss_episodes("https://x.com/feed.rss", quiet=True)
        assert exc.value.code == 3

    def test_network_error_exits_3(self):
        mock_urlopen = MagicMock(side_effect=OSError("Connection refused"))
        with patch("urllib.request.urlopen", mock_urlopen):
            with pytest.raises(SystemExit) as exc:
                fetch_rss_episodes("https://x.com/feed.rss", quiet=True)
        assert exc.value.code == 3

    def test_malformed_xml_exits_3(self):
        with patch("urllib.request.urlopen", _mock_urlopen(b"<not valid xml???")):
            with pytest.raises(SystemExit) as exc:
                fetch_rss_episodes("https://x.com/feed.rss", quiet=True)
        assert exc.value.code == 3

    def test_no_items_exits_3(self):
        empty_rss = b"""<?xml version="1.0"?>
        <rss version="2.0"><channel></channel></rss>"""
        with patch("urllib.request.urlopen", _mock_urlopen(empty_rss)):
            with pytest.raises(SystemExit) as exc:
                fetch_rss_episodes("https://x.com/feed.rss", quiet=True)
        assert exc.value.code == 3
