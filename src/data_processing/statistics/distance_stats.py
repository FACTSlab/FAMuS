"""
distance statistics: 
distribution of the distance between the first role filler and the last role filler in report and source docs. Distance is computed as the number of tokens.
tokens between the first filled roles and last filled role in report/source.
"""

import os 
import sys
import json
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm


