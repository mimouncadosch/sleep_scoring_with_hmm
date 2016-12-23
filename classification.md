Improving Classification accuracy
=================================

1/8/2016

1) Initial attempt: simple: √

2) Second attempt: increase signals used (added EOG): √

Lessons: 
Need better features, otherwise classification is worse than random.
Need better features to distinguish everything from wakefulness.

Need signals that enable to distinguish NREM and REM from wakefulness.

3) Third attempt: add more features (especially in the frequency domain): √

Lessons: 
Accuracy increased from 32% to 74% when adding EMG. 
Better features don't work if they don't come from the right data.

4) More sophisticated way of adding signals: √

Lessons:
This makes my life easier in testing different signals
Less signals produce better data, e.g. [5,7] produce better results than [5,7,8]

5) Compute more sophisticated statistics (precision / recall): √
Lessons:
Computing more precision and recall for each state was very insightful, as it confirmed my concern that the model would not be very good at discerning REM from NREM sleep. 

6) Analyze emission distributions

7) Train model with more data

8) PCA to determine which features to use

###Two approaches: increase features, and increase data

### Increase data:

### Download data: 