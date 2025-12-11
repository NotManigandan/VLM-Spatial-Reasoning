import json
import re
from pathlib import Path

COLOR_LIST_FILE = Path("/home/ubuntu/dataset/OmniSpatial-test/color_questions.txt")
TARGET_JSON = Path("/home/ubuntu/dataset/OmniSpatial-test/no_color_data.json")

def load_color_questions(path: Path):
    questions = set()
    if not path.exists():
        raise SystemExit(f"Color list not found: {path}")
    
    # Extract the questions.
    pat = re.compile(r'"question"\s*:\s*"(.*)"\s*,?\s*$', re.IGNORECASE)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = pat.search(line.rstrip("\n"))
            if m:
                questions.add(m.group(1))
    return questions

def main():
    color_qs = load_color_questions(COLOR_LIST_FILE)
    if not TARGET_JSON.exists():
        raise SystemExit(f"Target JSON not found: {TARGET_JSON}")

    with TARGET_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "data" in data:
        items = data["data"]
        wrapper = True
    else:
        items = data
        wrapper = False

    filtered = []
    removed_items = []
    for item in items:
        q = item.get("question") if isinstance(item, dict) else None
        if q is None:
            filtered.append(item)
            continue
        if q in color_qs:
            removed_items.append(item)
        else:
            filtered.append(item)

    # Write back
    with TARGET_JSON.open("w", encoding="utf-8") as f:
        if wrapper:
            json.dump({"data": filtered}, f, ensure_ascii=False, indent=2)
        else:
            json.dump(filtered, f, ensure_ascii=False, indent=2)

    # Write a file containing removed questions for audit
    diff_path = TARGET_JSON.with_name("removed_color_questions.txt")
    with diff_path.open("w", encoding="utf-8") as df:
        for item in removed_items:
            q = item.get("question") if isinstance(item, dict) else ""
            df.write(q + "\n")

    print(f"Removed {len(removed_items)} questions; kept {len(filtered)} items in {TARGET_JSON}")
    print(f"Wrote removed questions to {diff_path}")

if __name__ == "__main__":
    main()
