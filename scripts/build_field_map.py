"""Verify the spec's FDIC field candidates against the official dictionary and write
``config/fields.yaml``.

Usage: ``uv run python scripts/build_field_map.py [--force-download]``

Steps: download ``risview_properties.yaml`` (skipped when cached), check every candidate
code, resolve the spec's "find code" items by keyword search, write the map, and print a
report of codes that were not found or ambiguous.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx

from bankcanary.config import load_settings
from bankcanary.fields import (
    CANDIDATES,
    DEFAULT_DICTIONARY_PATH,
    DEFAULT_FIELD_MAP_PATH,
    KEYWORD_SEARCHES,
    LEGACY_CANDIDATES,
    build_field_map,
    parse_dictionary,
    write_field_map,
)

DICTIONARY_NAME = "risview_properties"


def download_dictionary(path: Path, force: bool = False) -> Path:
    """Fetch the financials dictionary from the FDIC docs URL unless it is already cached."""
    if path.exists() and not force:
        print(f"dictionary cached at {path}")
        return path
    settings = load_settings()
    url = f"{settings.fdic.docs_url}{DICTIONARY_NAME}.yaml"
    print(f"downloading {url}")
    response = httpx.get(url, timeout=settings.fdic.timeout_seconds, follow_redirects=True)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_FIELD_MAP_PATH)
    args = parser.parse_args(argv)

    dictionary = parse_dictionary(download_dictionary(args.dictionary, args.force_download))
    print(f"dictionary has {len(dictionary)} codes\n")

    print("== candidate check ==")
    for cand in (*CANDIDATES, *LEGACY_CANDIDATES):
        entry = dictionary.get(cand.code)
        status = "found    " if entry else "NOT FOUND"
        print(f"  {status} {cand.code:10} {(entry or {}).get('title', '')}")

    result = build_field_map(dictionary)
    print("\n== keyword searches ==")
    for search in KEYWORD_SEARCHES:
        print(f"  {search.label} (pattern {search.pattern!r}) -> chosen {search.chosen}")
        for code, title in result.search_report.get(search.label, []):
            marker = "*" if code == search.chosen else " "
            print(f"    {marker} {code:10} {title}")

    write_field_map(result, args.out)
    print(f"\nwrote {len(result.fields)} fields to {args.out}")
    print("\n== report ==")
    print("not found:", ", ".join(result.not_found) or "none")
    for note in result.notes:
        print("note:", note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
