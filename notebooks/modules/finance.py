import pandas as pd
import nsepy
import datetime
from   io import StringIO, BytesIO, TextIOWrapper
import zipfile, csv
import subprocess
import shlex
from   .utils import *

# NSE bhavcopy data
def nse_latest_bhavcopy_info(incl_series=['EQ']):
    l_file = None
    date_y   = datetime.datetime.today() - datetime.timedelta(days=1)    # yesterday date
    shift    = datetime.timedelta(max(1,(date_y.weekday() + 6) % 7 - 3))
    date_y   = date_y - shift
    month_y  = month_str(date_y.month)
    day_y    = date_y.day
    year_y   = date_y.year
    #url_this = "http://www.bseindia.com/download/BhavCopy/Equity/EQ{:02d}{:02d}{:02d}_CSV.ZIP".format(date_y.day, date_y.month, date_y.year % 2000)
    url_this  = "https://www1.nseindia.com/content/historical/EQUITIES/{}/{}/cm{:02d}{}{}bhav.csv.zip".format(year_y, month_y, day_y, month_y, year_y)
    print("Fetching NSE Bhavcopy from {}".format(url_this))
    #print "Fetching BSE Bhavcopy from {}".format(url_this)
    u_req    = Request(url_this)
    u_req.add_header('User-agent', 'Mozilla 5.10')
    d_data   = urlopen(u_req)
    l_file   = BytesIO(d_data.read())
    
    # Read zip file
    zip_f    = zipfile.ZipFile(l_file)
    items_f  = TextIOWrapper(zip_f.open(zip_f.namelist()[0]))
    csv_f    = csv.reader(items_f)
    nse_dict = {}
    ctr      = 0

    for item_this in csv_f:
        if ctr != 0:
            # Convert strings to integers/floats
            cc_dict = {
                "nse_code"        : item_this[0],
                "nse_group"       : item_this[1].rstrip(),
                "open"            : float(item_this[2]),
                "high"            : float(item_this[3]),
                "low"             : float(item_this[4]),
                "close"           : float(item_this[5]),
                "last"            : float(item_this[6]),
                "prev_close"      : float(item_this[7]),
                "no_trades"       : int(item_this[11]),
                "isin"            : item_this[12].rstrip(),
            }

            if incl_series != None:
                if item_this[1] in incl_series:
                    nse_dict[cc_dict['nse_code']] = cc_dict
                # endif
            else:
                nse_dict[cc_dict['nse_code']] = cc_dict
            # endif
        # endif
        ctr = ctr + 1
    # endfor
    return nse_dict
# enddef

##########################################################
def download_nse_daily_data(out_file, sym_list=None, start_date=None, end_date=None):
    start_date = datetime.datetime(1992, 1, 1) if start_date is None else start_date
    end_date   = datetime.datetime.now() if end_date is None else end_date
    sym_list   = list(nse_latest_bhavcopy_info().keys()) if sym_list is None else sym_list

    data_dict  = {}
    for indx_t, sym_t in enumerate(sym_list):
        if chkfile(out_file):
            data_dict = load_pickle(out_file)
            if sym_t in data_dict.keys():
                log_console('Symbol {} already downloaded !!'.format(sym_t))
                continue
            # endif
        # endif
        try:
            df_t = nsepy.get_history(sym_t, start_date, end_date)
            # We only want ohlcv data
            df_t = df_t[['Open', 'High', 'Low', 'Close', 'Volume']]
            data_dict[sym_t] = df_t
            log_counter_console(indx_t+1, len(sym_list), sym_t)
            save_pickle(data_dict, out_file)
        except:
            log_console('Error encountered while fetching {}'.format(sym_t))
            pass
        # endtry
    # endfor
# enddef

# Using Gnuplot
def generate_candlestick_figure(df_t, out_file,
        col_map={'o':'Open', 'h':'High', 'l':'Low', 'c':'Close', 'v': 'Volume'}):
    date_l   = df_t.index.to_list()
    open_l   = df_t[col_map['o']].to_list()
    high_l   = df_t[col_map['h']].to_list()
    low_l    = df_t[col_map['l']].to_list()
    close_l  = df_t[col_map['c']].to_list()

    file_ext = os.path.splitext(out_file)[1]
    file_ext = file_ext[1:].lower() if file_ext[0] == '.' else 'png'
    assert file_ext in ['png', 'jpg'], '{} should have a png or jpg file extension'.format(out_file)

    gpl_txt  = '''
reset
set key off
set obj 1 rectangle behind from screen 0,0 to screen 1,1
set obj 1 fillstyle solid 1.0 fillcolor rgbcolor "black"

set xdata time
set timefmt "%Y-%m-%d"
set datafile separator ","
set palette defined (-1 'red', 1 'green')
set cbrange [-1:1]
unset colorbox
set style fill solid noborder
set boxwidth 30000 absolute
set bars linecolor "white" linewidth 1
set term {}
set output "{}"

plot "-" using (strptime("%Y-%m-%d", strcol(1))):2:4:3:5:($5 < $2 ? -1 : 1) with candlesticks palette'''.format(file_ext, out_file)

    # Add rows
    for i in range(len(date_l)):
        gpl_txt += '\n{},{},{},{},{}'.format(date_l[i], open_l[i], high_l[i], low_l[i], close_l[i])
    # endfor

    proc = subprocess.Popen(['gnuplot'], shell=False, stdin=subprocess.PIPE)
    proc.stdin.write(gpl_txt.encode('utf-8'))
    stdout, stderr = proc.communicate()
    if stdout:
        console_log(stdout)
    if stderr:
        console_log(stderr)
    # endif
# enddef
