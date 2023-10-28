"""
role sets between report and source
2.1 What do the role sets look like between report and source passages?
2.2 Is the report often a subset/superset/exact match of the source roles? -- it feels like it should be an exact match, but sometimes it isn't... why?
2.3 When the event trigger is in a quote from the source, how often does the report contain additional/less roles?
"""

import os 
import json
import argparse
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from famus.src.