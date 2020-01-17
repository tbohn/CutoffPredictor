#!/usr/bin/env/python

import argparse
import os.path
from backend import process_wrapper as prwr

# -------------------------------------------------------------------- #
def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='CutoffPredictor '
                                     'uses machine learning to identify '
                                     'utility customers at risk of '
                                     'service interruption.')
    parser.add_argument('config_file', type=str,
                        help='Input configuration file',
                        default=None, nargs='?')
    parser.add_argument('--version', action='store_true',
                        help='Return CutoffPredictor version string')

    args = parser.parse_args()

    if args.version:
        from backend import version
        print(version.short_version)
        return

    if (args.config_file):
        if not os.path.isfile(args.config_file):
            raise IOError('config_file: {0} does not '
                          'exist'.format(args.config_file))
        else:
            prwr.process_wrapper(args.config_file)
    else:
        parser.print_help()

    return
# -------------------------------------------------------------------- #

# -------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
# -------------------------------------------------------------------- #
