import os
import pandas as pd
import pickle
from   urllib.request import urlopen, Request
import logging
import urllib3
import subprocess
import shlex

# Disable pesky urllib3 warnings
logging.getLogger(urllib3.__package__).setLevel(logging.ERROR)

###########################################################################################
# File system handling

def mkdir(dir):
    if dir == None:
        return None
    # endif
    if not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=True)
    # endif
# enddef

def rp(dir):
    if dir == None:
        return None
    # endif
    if dir[0] == '.':
        return os.path.normpath(os.getcwd() + '/' + dir)
    else:
        return os.path.normpath(os.path.expanduser(dir))
# enddef

def filename(x, only_name=True):
    n_tok = os.path.splitext(os.path.basename(x))
    return n_tok[0] if only_name else n_tok
# enddef

def chkdir(dir):
    if not os.path.isdir(dir):
        print('{} does not exist !!'.format(dir))
        sys.exit(-1)
    # endif
# enddef

def chkfile(file_path, exit=False):
    if os.path.exists(file_path) and os.path.isfile(file_path):
        return True
    # endif
    if exit:
        print('{} does not exist !!'.format(file_path))
        sys.exit(-1)
    # endif
        
    return False
# enddef

def npath(path):
    return os.path.normpath(path) if path else None
# enddef

#################################################
# Helpers

def load_json(json_file):
    return json.load(open(json_file, 'r'))
# enddef

def save_json(json_obj, json_file, indent=4):
    json.dump(json_obj, open(json_file, 'w'), indent=indent)
# enddef

def load_pickle(pickle_file):
    return pickle.load(open(pickle_file, 'rb')) if os.path.isfile(pickle_file) else None
# enddef

def save_pickle(pickle_obj, pickle_file):
    pickle.dump(pickle_obj, open(pickle_file, 'wb'))
# enddef

def log_console(msg):
    print('>> {}'.format(msg))
# enddef

def log_counter_console(curr_ctr, total_ctr, ext_msg=None):
    if ext_msg:
        print('>> [{:<4}/{:<4}] -> {}'.format(curr_ctr, total_ctr, ext_msg), end='\r')
    else:
        print('>> [{:<4}/{:<4}]'.format(curr_ctr, total_ctr), end='\r')
    # endif
# enddef

def month_str(m_n):
    assert m_n <= 12 and m_n >= 1, "Month number should be between 1 & 12"
    m_str = [ "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC" ]
    return m_str[m_n-1]
# enddef
