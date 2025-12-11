import re
from pathlib import Path

DATA_FILE = Path("/home/ubuntu/dataset/OmniSpatial-test/data.json")
OUT_FILE = Path("/home/ubuntu/dataset/OmniSpatial-test/color_questions.txt")

COLOR_TERMS = [
    "color", "colour", "red", "blue", "green", "yellow", "orange", "purple",
    "pink", "brown", "black", "white", "gray", "grey", "cyan", "magenta",
    "beige", "maroon", "navy", "teal", "violet", "indigo", "gold", "silver",
]

# Build a regex that matches color terms as whole words or hyphenated compounds,
# avoiding substrings inside longer words (e.g., "captured" matching "red").
# Examples allowed: "red", "red-wrapped", "white-box"; disallowed: "captured".
term_patterns = []
for t in COLOR_TERMS:
    # Word boundary or start of string or non-letter before; then the term;
    # then word boundary or end of string or non-letter after.
    term_patterns.append(rf"(?<![A-Za-z]){re.escape(t)}(?![A-Za-z])")

COLOR_REGEX = "|".join(term_patterns)

# Match a JSON line that contains a question string and any color term
PATTERN = re.compile(r"\"question\"\s*:\s*\"[^\"]*(?:" + COLOR_REGEX + r")[^\"]*\"", re.IGNORECASE)


def main():
    if not DATA_FILE.exists():
        raise SystemExit(f"Data file not found: {DATA_FILE}")

    matches = []
    seen = set()

    with DATA_FILE.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if PATTERN.search(line):
                text = line.rstrip("\n")
                key = (idx, text)
                if key not in seen:
                    seen.add(key)
                    matches.append((idx, text))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as out:
        for ln, txt in matches:
            out.write(f"line {ln}: {txt}\n")

    print(f"Wrote {len(matches)} color-related questions to {OUT_FILE}")


if __name__ == "__main__":
    main()
