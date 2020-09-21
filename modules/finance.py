import pandas as pd
import nsepy
import datetime
from   io import StringIO, BytesIO, TextIOWrapper
import zipfile, csv
import subprocess
import shlex
import numpy as np
import random
import mplfinance as mplf
from   decimal import Decimal
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import PIL.Image
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

def format_price(x, _=None):
    x = Decimal(x)
    return x.quantize(Decimal(1)) if x == x.to_integral() else x.normalize()
# enddef

# Plot candlestick to out_file if out_file is not None,
# else return a PIL instance for the image
def generate_candlestick_figure(df, out_file=None, figratio=None, figscale=1.0, return_pil=False):
    df.index = pd.to_datetime(df.index)
    # Generate log plot. Currently it's not supported by mplf.plot,
    # so we do it overselves.
    if figratio:
        fig, axlist = mplf.plot(df, type='candle', returnfig=True, figratio=figratio, figscale=figscale)
    else:
        fig, axlist = mplf.plot(df, type='candle', returnfig=True, figscale=figscale)
    # endif
    ax1 = axlist[0]
    ax1_minor_yticks = ax1.get_yticks(True)  # save the original ticks because the log ticks are sparse
    ax1_major_yticks = ax1.get_yticks(False)
    ax1.set_yscale('log')
    ax1.set_yticks(ax1_major_yticks, True)
    ax1.set_yticks(ax1_minor_yticks, False)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(format_price))
    ax1.yaxis.set_minor_formatter(ticker.FuncFormatter(format_price))
    ax1.autoscale()
    ax1.axis('off')

    buf = None
    if out_file:
        fig.savefig(out_file)
    else:
        buf = BytesIO()
        fig.savefig(buf, format='png')
        buf = PIL.Image.open(buf) if return_pil else buf
    # endif
    plt.close('all')
    return buf
# enddef

# Generate training data from ticker
def generate_training_data_from_ticker(df_t, out_dir, num_samples=200,
        forward_period=4, backward_period=20, tick_name=None,
        timeout=40, figratio=None, figscale=1.0):
    mkdir(out_dir)
    
    tick_name = u4() if tick_name is None else tick_name
    df_t['return'] = np.log(df_t['Close']) - np.log(df_t['Close'].shift())
    df_t['cumret'] = df_t['return'].rolling(window=forward_period).sum().shift(-forward_period)
    df_t.dropna(inplace=True)
    sample_index_list = []
    annot_dict = {}
    tout_ctri = 0

    if backward_period >= len(df_t)-1:
        return {}
    # endif

    for indx_t in range(num_samples):
        if tout_ctri >= timeout:
            break
        # endif
        tout_ctr   = 0
        # Select starting index
        while True:
            if tout_ctr >= timeout:
                break
            # endif
            start_indx = random.randint(backward_period, len(df_t)-1)
            if start_indx in sample_index_list:
                tout_ctr += 1
                continue
            # endif
            break
        # endwhile
        if start_indx in sample_index_list:
            tout_ctri += 1
            continue
        # endif
        
        # Append
        sample_index_list.append(start_indx)
        
        # Generate data
        cumret_t = df_t.iloc[start_indx]['cumret']
        cumr_str = ('plus_{}'.format(cumret_t) if cumret_t > 0 else 'minus_{}'.format(abs(cumret_t))).replace('.', '')
        img_out  = tick_name + '_' + str(indx_t) + '_' + str(start_indx) + \
                       '_' + str(df_t.iloc[start_indx].name) + '_' + cumr_str + '.png'
        img_out_abs = os.path.join(out_dir, img_out)
        annot_dict[img_out] = df_t.iloc[start_indx]['cumret']
        
        generate_candlestick_figure(df_t.iloc[start_indx+1-backward_period:start_indx+1], img_out_abs,
                                    figratio=figratio, figscale=figscale)
    # endfor
    return annot_dict
# enddef

# Generate training data from ticker
def dframe_to_tfdata(df_t, forward_period=4, backward_period=20, figratio=None, figscale=1.0):
    df_t['return'] = np.log(df_t['Close']) - np.log(df_t['Close'].shift())
    df_t['cumret'] = df_t['return'].rolling(window=forward_period).sum().shift(-forward_period)
    df_t.dropna(inplace=True)
        
    # Generate data
    start_indx = random.randint(backward_period, len(df_t)-1)
    cumret_t   = df_t.iloc[start_indx]['cumret']
        
    data = generate_candlestick_figure(df_t.iloc[start_indx+1-backward_period:start_indx+1], figratio=figratio, figscale=figscale)
    return (data, cumret_t)
# enddef
