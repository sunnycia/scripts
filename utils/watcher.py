import os, glob
import time
map_base = '/data/sunnycia/SaliencyDataset/Video/Coutort2/saliency_map'

while True:
    time.sleep(5)
    xu_len = len(glob.glob(os.path.join(map_base, 'xu_lstm')))
    pqft_len = len(glob.glob(os.path.join(map_base, 'pqft')))
    3dres_len = len(glob.glob(os.path.join(map_base, 'vo-v4-2-resnet-dropout-snapshot-2000-display-1-dropout_fulldens-batch-2_1514857787_snapshot-_iter_26000_threshold0')))

    print xu_len, pqft_len, 3dres_len;exit()