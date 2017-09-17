from __future__ import print_function

import os
import sys


def list_all_files(directory, file_ext):
    for root, dirs, files in os.walk(directory):
        for fname in files:
            if fname.endswith(file_ext):
                yield os.path.join(root, fname)

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('--directory', '-d', default=os.getcwd())
    parser.add_argument('--extension', '-e', default='.c')
    parser.add_argument('--output', '-o', default=sys.stdout)
    args = parser.parse_args()

    charset = set()
    with open(args.output, 'w') as output:
        for fname in list_all_files(args.directory,
                                    args.extension):
            with open(fname) as fstream:
                for line in fstream:
                    print(line.rstrip(), file=output)
                    charset = charset.union(set(line))
                # Add an extra line for formatting
                print ('', file=output)
                
    print ('unique characters used:', len(charset))

