import pandas as pd
from utils import io_util

labels = pd.read_csv('TransTVDiag/data/gaia/label_15_85.csv')
data = {idx: {} for idx in labels['index'].values}
io_util.save('/GAIA_data/data/gaia/pkl/gaia.pkl', data)
