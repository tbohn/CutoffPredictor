#!/usr/bin/env/python

import argparse
import os.path
from backend import config as cfg
from dashboard import app

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
            # Read config_file
            if isinstance(args.config_file, dict):
                config = args.config_file
            else:
                config = cfg.read_config(args.config_file)
            # Start the dashboard
            app.dashboard(config)
    else:
        parser.print_help()

    return
# -------------------------------------------------------------------- #

# -------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
# -------------------------------------------------------------------- #
