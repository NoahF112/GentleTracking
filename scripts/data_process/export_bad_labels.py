import argparse
import json
import os


def _is_bad_record(rec):
    status = rec.get("status")
    if status is not None:
        return str(status).lower() in {"bad", "abnormal", "1", "true"}
    if "is_bad" in rec:
        return bool(rec["is_bad"])
    if "label" in rec:
        return str(rec["label"]).lower() in {"bad", "abnormal", "1", "true"}
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flagged",
        default="/home/axell/Desktop/gmt/outputs/flagged_motion_ids.jsonl",
        help="JSONL file with motion_id and status",
    )
    parser.add_argument(
        "--id-label",
        default="/home/axell/Desktop/gmt/dataset/amass_all_0111/id_label.json",
        help="id_label.json from dataset",
    )
    parser.add_argument(
        "--output",
        default="/home/axell/Desktop/gmt/outputs/label.txt",
        help="Output label.txt path",
    )
    args = parser.parse_args()

    with open(args.id_label, "r", encoding="utf-8") as f:
        id_labels = json.load(f)

    bad_ids = set()
    with open(args.flagged, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not _is_bad_record(rec):
                continue
            motion_id = rec.get("motion_id")
            if motion_id is None:
                continue
            bad_ids.add(int(motion_id))

    entries = []
    for motion_id in sorted(bad_ids):
        if motion_id < 0 or motion_id >= len(id_labels):
            continue
        label = id_labels[motion_id]
        entries.append(
            f"{label.get('source_path','')}\t{label.get('segment_start','')}\t{label.get('segment_end','')}"
        )

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        for line in entries:
            f.write(line + "\n")

    print(f"Wrote {len(entries)} labels to {args.output}")


if __name__ == "__main__":
    main()
