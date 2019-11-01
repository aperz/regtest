# regtest

This script performs pariwise comparisons between variables in a data frame
and saves some plots for the most extreme relationships with regard to a
metric, e.g. the mean squared difference.

It's by no means formal hypothesis testing, but should give some intuition of how the variables
in the data frame are related to each other.

### Usage:
$ python3 regtest.py -h

### TODO:
- replace this permutation ridiculousness with an actual distribution.
- implement a solver based on polyfit and classify relationship as linear if
if best fit is with d=1
