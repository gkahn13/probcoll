import numpy as np


def make_immutable(*args):
    for a in args:
        if not isinstance(a, object): continue
        if isinstance(a, np.ndarray):
            a.setflags(write=False)
        # elif TODO_TZ: other types?

# del np