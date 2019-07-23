# Configuration File

## Formatting of the file

The format of the configuration file is:
    [SECTION1]
    KEY1 : value1
    KEY2 : value2
    ...
    
    [SECTION2]
    ...

CutoffPredictor will store this information in a dictionary called `config`, which can be accessed as follows:
    `config[SECTION1][KEY1]` gives `value1`

Keys defined on one line of the config file can be referenced on subsequent lines via `%(KEY)s`, where `KEY` is the key. I.e., `%(KEY1)s` will be interpreted as `value1`.

## Major key-value options

The keys that need to be defined are:
1. `[PATHS]`:

