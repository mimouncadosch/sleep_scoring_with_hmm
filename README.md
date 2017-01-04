## Automatic Sleep Staging using Hidden Markov Models

### Intro
Sleep staging consists of taking data from a polysomnography and labeling the
time intervals according to the dierent possible sleep stages (1, 2, 3, 4 and
REM). Sleep staging is an integral part of sleep medicine, as it enables special-
ists to diagnose dierent types of sleep disorders.

### Pain point
The process of sleep staging is usually done by hand by a qualied technician,
taking anywhere from two to three hours depending on the technician's level of
expertise and care. Sleep staging can be viewed as a classication problem. As
such, it is one that can benet from machine learning for a faster and thus less
expensive solution.

### Machine learning as a solution
Researchers have studied automated sleep staging in the past [1], and some
have even commercialized automated sleep staging systems [2]. These systems
perform classication, however, based on sets of heuristic rules rather than on
supervised machine learning techniques.

### Choices
The particular model we have decided to use are Hidden Markov Models. This
model seems suitable because in this case, the hidden states are the sleep stages,
while the visible states are the dierent signal values. Additionally, this model
seems suitable for the problem at hand given the time-dependent nature of the
sleep stages. That is to say, the probability of an interval corresponding to a
given stage, is dependent on the sleep stage of the previous interval. For in-
stance, a patient is much likelier to transition from stage 3 to stage 4 than he
is from stage 3 to stage 2, as humans tend to follow these stages in their given
order.


#### Project references
See `plan.md` and `classification.md`.


#### Sources
```
[1] Performance of an Automated Polysomnography Scoring System Versus
Computer-Assisted Manual Scoring, Malhotra et al., SLEEP, 2012

[2] Michele Sleep Scoring https://michelesleepscoring.com/
```
