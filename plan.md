Plan
========

### Caching Arrays: √
* Cache all the signals in RAM so you can access them more quickly: √
* In the future, you can try more sophisticated methods such as caching arrays of a given (smaller) size, and returning the sub-arrays when requested.
* Then, if datapoints in a different range are requested, caching arrays in such range.
* This ensures fast retrieval of nearby datapoints, while some waitgfor retrieving datapoints in far range.
* Retrieving array from disk is 7 orders of magnitude longer retrieving from RAM.

### Get 30 seconds worth of each signal: √

### Plot signal: √

### Slider to change epoch: √

### Define and extract features:  WIP
* Features to extract for a given epoch:
	1. Mean, stdev, skewness, kurtosis, max, min: √
	2. Autocorrelation, AR, MA: TODO

### Isolate sleep times in data: √
* SC4001E0: ei: 1,000, ne: 850

### Open 20000* with python-EDF: √
* Lesson learned: use sys.argv, do not hard code the filename, as you can make typos.
* Figure out why can't use shhs1 data with python: √

### Retrieve annotations from CSV (staging-csv): √

<!-- SAT -->
### Hidden Markov Model:
* Learn how to use hmmlearn library: √
* Should I learn the parameters? Or get transition matrix otherwise
* Use SVM for emission probabilities

### Compute transition probabilities: √

### Use simple multi-class SVM to see if you can label without HMM:√
* Spoke with Ethan, he helped me understand what I need to do

### Average transition probabilities for many datasets:

### Remove noise / clean data:
* Observations from data:
	1. 2*2: 
		- When awake, the frequency of EMG, EOG, and Air flow are very high
		- During N2 - N3 (NREM), frequencies are not as high, seem normal
		- It seems like the classifier might classify very high frequencies and amplitudes as sleep. 
		- *** make sure that amplitudes are part of the features ***
	2. 2*3: 
		- Best predictor of wakefulness is EMG
		- When awake, EMG has high frequency and high amplitudes. 
		- EOG is non-periodic, has high frequency and high amplitudes
		- In NREM, EMG has much lower amplitudes. EOG has much lower amplitudes
		- During REM, EOG has very low frequencies with high amplitudes, and is non-periodic. 
		- *** make sure that periodicity is part of the features ***
	

### Estimate transition probabilities: save matrix: √

### Estimate emission probabilities: √

### Save emission probabilities in npz file: √

### Test and score model: √

### Save parameters as single .npy file: √

### Improve classification accuracy [ see classification.md ] 

### Have transition probabilities change with time



Plan:
JAN
HMM -> classify REM, NREM using breathing and heart
Kinect -> get breathing and try to get heart
FEB
CBT-I -> determine what you are going to do
Finalize parts that go into the bedboard
Determine design by hand, hire industrial designer