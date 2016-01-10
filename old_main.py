import os
import edflib
import matplotlib.pyplot as plt
import numpy as np

from biosignal import Signal
from epoch import get_epoch_data, get_epoch_times, number_of_epochs
from dataio import load_signals, get_signal_freqs
from features import autocorr, get_features

#########################
####### Main.py #########
#########################
# fname = './data/SC4001E0/SC4001E0-PSG.edf'
fname = './data/edfs/shhs1-200002.edf'
e = edflib.EdfReader(fname)
signal_indices = [7, 8]
eeg1, resp = load_signals(e, signal_indices)
# freqs = get_signal_freqs(e)

eeg1_f = 125; resp_f = 10
ei = 0; ne = 1

print "File name: %s" %fname
print "Signal labels: ", e.getSignalTextLabels()
print "Initial ei:%d, ne:%d" %(ei, ne)
print "Number of epochs: %d" %number_of_epochs(eeg1, eeg1_f)

# 1) Raw data
data_resp = get_epoch_data(resp, resp_f, ei, ne)
times_resp = get_epoch_times(ei, ne, resp_f)

data_eeg = get_epoch_data(eeg1, eeg1_f, ei, ne)
times_eeg = get_epoch_times(ei, ne, eeg1_f)

# 2) Transform to frequency domain
s_resp = Signal(data_resp, resp_f, 0)
sf_resp = s_resp.to_freq_domain()

s_eeg = Signal(data_eeg, eeg1_f, 0)
sf_eeg = s_eeg.to_freq_domain()

# 3) Apply low pass filter
sf_resp_copy = sf_resp.copy()
sf_eeg_copy  = sf_eeg.copy()

sf_resp.low_pass(0.3, factor=0)
sf_eeg.low_pass(0.5, factor=0)

# 4) Recreate signal from filtered spectrum
resp_filtered 	= sf_resp.to_time_domain()
eeg_filtered 	= sf_eeg.to_time_domain()



# 5) Retrieve features in the time domain
resp_mean, resp_var, resp_skew, resp_kurt, resp_max, resp_min = get_features(resp_filtered.ys)
resp_autocorr = autocorr(resp_filtered.ys)

resp_string = 	"""
				$\mu_{filtered} = %f$ \n
				$\sigma_{filtered} = %f$ \n
				$\gamma_{filtered} = %f$ \n
				$\kappa_{filtered} = %f$ \n"""% (resp_mean, resp_var, resp_skew, resp_kurt)

# Plot signal
fig = plt.figure( num=0, figsize=(12, 12), dpi=100 )
fig.suptitle("Bedboad Sleep Analysis v1", fontsize=13)

ax00 = plt.subplot2grid( (4,4), (0,0) )
ax01 = plt.subplot2grid( (4,4), (0,1) )
ax02 = plt.subplot2grid( (4,4), (0,2) )
ax03 = plt.subplot2grid( (4,4), (0,3) )

ax10 = plt.subplot2grid( (3,3), (1,0) )
ax11 = plt.subplot2grid( (3,3), (1,1) )
ax12 = plt.subplot2grid( (3,3), (1,2) )
# ax13 = plt.subplot2grid( (3,3), (1,3) )

ax00.plot(times_resp, data_resp)
ax01.plot(sf_resp_copy.fs, sf_resp_copy.amps)
ax02.plot(times_resp, resp_filtered.ys)
ax03.annotate(resp_string, (0, 0), textcoords='axes fraction', size=10)
# ax03.text(0, 0.8, resp_string)


ax10.plot(times_eeg, data_eeg)
ax11.plot(sf_eeg_copy.fs, sf_eeg_copy.amps)
ax12.plot(times_eeg, eeg_filtered.ys)
# ax13.text('Hello world')

plt.show()
