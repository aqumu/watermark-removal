import random
import numpy as np

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)