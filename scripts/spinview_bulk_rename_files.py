import argparse
from pathlib import Path
import csv
import sys
import re
import json
from datetime import datetime, timedelta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--indexlength', type=int, default=4)
    parser.add_argument('-c', '--dry', action='store_true')
    parser.add_argument('glob', type=str)
    parser.add_argument('csv_out', type=str)
    parser.add_argument('summary_out', type=str)
    params = parser.parse_args()

    paths = sorted(Path('.').glob(params.glob))

    if params.dry:
        csv_file = sys.stdout
    else:
        if params.csv_out == '-':
            csv_file = sys.stdout
        else:
            csv_file = open(params.csv_out, mode='a', newline='')
    csv_writer = csv.writer(csv_file)

    pattern = re.compile(r"^(\d+)\-(\d{8}(\d{4})\d{2})\-(\d+)$")
    summary = dict()

    timedelta_threshold = timedelta(seconds=5)
    previous_capture_group = None
    previous_capture_time = None

    if params.dry:
        new_paths = set()
        duplicate_new_paths = set()

    for path in paths:
        path: Path
        match = pattern.match(path.stem)
        if not match:
            continue
        new_index = match.group(4).zfill(params.indexlength)
        new_suffix = path.suffix.lower()

        capture_group = match.group(1)
        capture_time = datetime.strptime(match.group(2), '%m%d%Y%H%M%S')
        if previous_capture_group != capture_group:
            capture_subgroup = 0
            previous_capture_group = capture_group
            current_summary = {
                'minutes': set(),
            }
            summary[capture_group] = current_summary
        else:
            # If too long has passed since the previous capture, we are in a new capture_subgroup
            time_diff = capture_time - previous_capture_time
            if time_diff >= timedelta_threshold:
                capture_subgroup += 1
        previous_capture_time = capture_time

        capture_subgroup_str = str(capture_subgroup).zfill(2)
        csv_writer.writerow((capture_group, capture_subgroup_str, new_index, match.group(2)))
        new_path = path.with_name(capture_group + '-' + capture_subgroup_str + '_' + new_index + new_suffix)

        if params.dry:
            print(new_path)
            if new_path in new_paths:
                duplicate_new_paths.add(new_path)
            else:
                new_paths.add(new_path)
        else:
            path.rename(new_path)

        current_summary['minutes'].add(match.group(3))
        current_summary['last_subgroup'] = capture_subgroup_str

    for i, item in summary.items():
        item['minutes'] = sorted(item['minutes'])
    if params.dry:
        summary_file = sys.stdout
    else:
        summary_file = sys.stdout if params.summary_out == '-' else open(params.summary_out, mode='w')
    json.dump(summary, summary_file, indent=4)

    if params.dry and len(duplicate_new_paths) > 0:
        print(duplicate_new_paths)
        exit(1)


if __name__ == '__main__':
    main()
