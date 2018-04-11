import argparse
import signal

from multiprocessing import Pool


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def add_prefix_to_words(file_path, prefix, to_stdout=False):

    prefix = prefix+'_'
    file_path_splitted = file_path.split('.')
    path, lang = ".".join(file_path_splitted[:-1]), file_path_splitted[-1]
    save_path = path+'.prefixed.'+lang
    with open(file_path) as f, open(save_path, 'w') as output:
        for line in f:
            line = line.strip().split()
            if len(line) < 1:
                if to_stdout:
                    print()
                else:
                    output.write('\n')
                continue
            new_line = [prefix + line[0]]

            is_embedding = True if len(line) > 90 else False

            if not is_embedding:
                for token in line[1:]:
                    new_line.append(prefix + token)
            else:
                new_line.extend(line[1:])
            to_write = " ".join(new_line)
            if to_stdout:
                print(to_write)
            else:
                output.write(to_write + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpora",
                        nargs='+',
                        required=True,
                        help="The list of files to prepend with language information")
    parser.add_argument("--prefixes",
                        required=True,
                        nargs='+',
                        help="The list of prefix to add to the corpus files")
    parser.add_argument("--stdout",
                        action='store_true',
                        default=False,
                        help="Print results to stdout.")

    args = parser.parse_args()

    corpora = args.corpora
    prefixes = args.prefixes

    assert len(corpora) == len(prefixes), "The corpora and prefixes lists have to be the same length"

    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    p = Pool(len(corpora), init_worker)
    signal.signal(signal.SIGINT, original_sigint_handler)
    out = [args.stdout]*len(corpora)
    try:
        p.starmap(add_prefix_to_words, zip(corpora, prefixes, out))
    except KeyboardInterrupt:
        print('Parent process received ctrl-c')
        p.terminate()
    else:
        p.close()
if __name__ == '__main__':
    main()