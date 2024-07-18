# forecast indicators
from .forecast_metrics import Hit, Miss, FalseAlarm, CorrectNegative
import os
from glob import glob
import numpy as np
from typing import Optional

# threshold vals of precipitation levels
THRESHOLD = [80, 200, 350, 500]

def QoF(mask, pred_dir, gt_dir, saveto:Optional[str]=None)->None:
    '''
    Write out the Qualitative metrics of Forecasting as txt files.

    :param ndarray mask: to maskout on-sea values.
    :param str pred_dir: root path of model prediction.
    :param str gt_dir: root path of ground truth. Should be aligned with the "pred_dir".
    :param Optional[None] saveto: target path for saving the txt files.
    :return None:
    '''
    H, M, FA, CN = [], [], [], []
    DATE = []

    if saveto is None:
        saveto = os.path.join(pred_dir, 'QoFs_topo')
    if not os.path.exists(saveto):
        os.mkdir(saveto)

    val_pred_paths = glob(os.path.join(pred_dir, 'val_pred', '*.npy'))
    test_pred_paths = glob(os.path.join(pred_dir, 'test_pred', '*.npy'))

    def writeqoftxt(paths, datatype:str, saveto=saveto):
        flat_mask = mask.flatten()
        for thre in THRESHOLD:
            name = datatype + '_' + str(thre) + 'mm.txt' # e.g. val_80mm.txt
            with open(os.path.join(saveto, name), 'a') as f:
                f.write("Date, Hit, Miss, FA, CN")
                f.write('\n')
                for n, path in enumerate(paths):
                    pred = np.load(path).flatten()
                    date = path[-12:] # yyyymmdd.npy
                    gt = np.load(os.path.join(gt_dir, date)).flatten()

                    if np.max(gt)<thre:
                        # print("Date: ", date)
                        continue
                    else:
                        print(f"Exporting ... {date}")
                        f.write(date[:-4] + ',')
                        f.write(str(Hit(pred=pred, gt=gt, mask=flat_mask, threshold=thre))+ ',')
                        f.write(str(Miss(pred=pred, gt=gt, mask=flat_mask, threshold=thre))+ ',')
                        f.write(str(FalseAlarm(pred=pred, gt=gt, mask=flat_mask, threshold=thre))+ ',')
                        f.write(str(CorrectNegative(pred=pred, gt=gt, mask=flat_mask, threshold=thre)))
                        f.write('\n')
            
    writeqoftxt(paths=val_pred_paths, datatype='val')
    writeqoftxt(paths=test_pred_paths, datatype='test')
    print("Successfully write to QoF.txts")
    return

if __name__ == '__main__':
    print("Import pass.")