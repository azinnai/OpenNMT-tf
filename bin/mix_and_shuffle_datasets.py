import os
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",
                        nargs='+',
                        required=True,
                        help='A sequence of source corpora')
    parser.add_argument("--target",
                        nargs='+',
                        required=True,
                        help='A sequence of target corpora')
    parser.add_argument("--source_output",
                        required=True,
                        help='The output basename')
    parser.add_argument("--target_output",
                        required=True,
                        help='The output basename')

    args = parser.parse_args()

    if not any(os.path.isfile(file) for file in args.source+args.target):
        raise FileNotFoundError("Some files don't exist!")

    source_lines = []
    target_lines = []

    for source_file, target_file in zip(args.source, args.target):
        with open(source_file, 'r') as f:
            for line in f:
                source_lines.append(line.strip())
        with open(target_file, 'r') as f:
            for line in f:
                target_lines.append(line.strip())

    assert(len(target_lines) == len(source_lines))

    lines = list(range(len(source_lines)))
    np.random.shuffle(lines)

    with open(args.source_output, 'w') as source_f, open(args.target_output, 'w') as target_f:
        for step, index in enumerate(lines):
            source_f.write(source_lines[index]+'\n')
            target_f.write(target_lines[index]+'\n')

if __name__ == "__main__":
    main()
