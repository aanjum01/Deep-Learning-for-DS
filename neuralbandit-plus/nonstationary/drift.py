"""Page-Hinkley drop detector (plan phase-6c, R6): [gama2014, §3.2.1, App. A Alg. 1; page1954] via river 0.27.

River defaults kept (delta = 0.005, min_instances = 30; no source is given for them). Changed: mode "down" (drops only), alpha = 1
(Gama et al.'s formula has no fading factor), threshold = lambda from the boosted-share budget (plan step 5).
"""

from river.drift import PageHinkley


def make_detector(threshold):
    return PageHinkley(threshold=threshold, mode="down", alpha=1.0)
