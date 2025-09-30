import os
import re
from typing import List, Dict


def find_event_files(base_dirs: List[str]) -> List[str]:
    matches = []
    for d in base_dirs:
        d = os.path.expanduser(d)
        for root, _, files in os.walk(d):
            for fname in files:
                if fname.startswith('events'):
                    matches.append(os.path.join(root, fname))
    return sorted(list(set(matches)))


def parse_tags_from_path(path: str, regex_patterns: List[str]) -> Dict[str, str]:
    """Extract named groups from path using user-defined regex patterns."""
    md = {}
    for pattern in regex_patterns:
        try:
            match = re.search(pattern, path)
            if match:
                md.update({k:v for k,v in match.groupdict().items() if v is not None})
        except re.error:
            continue  # skip invalid patterns
    md['_path'] = path
    return md
