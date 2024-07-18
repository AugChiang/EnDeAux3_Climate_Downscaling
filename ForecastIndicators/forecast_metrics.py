# For the definition of metrics, ref:
# https://www.cawcr.gov.au/projects/verification/verif_web_page.html
import numpy as np

def Hit(pred, gt, mask, threshold)->int:
    '''
    Binarize predicted pixel values and ground truth pixel values with the thresholds.
    A "Hit" is counted when "event forecast to occur, and did occur".

    :param ndarray pred: model prediction.
    :param ndarray gt: ground truth.
    :param ndarray mask: used to maskout on-sea values. We focus values on land. Default mask location is at "../mask/mask_sd5km.npy".
    :param float threshold: is set according to the levels of raining amplitudes.
    :return int: number of counts that both predicted pixel values and ground truth pixel values are 1 (True).
    '''
    gt_hit = (gt>=threshold).astype(int)
    hit = (pred>=threshold).astype(int)
    hit = hit*gt_hit*mask
    return hit[hit>0].shape[0]

def Miss(pred, gt, mask, threshold)->int:
    '''
    Binarize predicted pixel values and ground truth pixel values with the thresholds.
    A "Miss" is counted when "event forecast NOT to occur, but did occur".

    :param ndarray pred: model prediction.
    :param ndarray gt: ground truth.
    :param ndarray mask: used to maskout on-sea values. We focus values on land. Default mask location is at "../mask/mask_sd5km.npy".
    :param float threshold: is set according to the levels of raining amplitudes.
    :return int: number of counts that both predicted pixel values and ground truth pixel values are 1 (True).
    '''
    gt_hit = (gt>=threshold).astype(int)
    miss = (pred<threshold).astype(int)
    miss = miss*gt_hit*mask
    return miss[miss>0].shape[0]

def FalseAlarm(pred, gt, mask, threshold)->int:
    '''
    Binarize predicted pixel values and ground truth pixel values with the thresholds.
    A "False Alarm" is counted when "event forecast to occur, but did NOT occur".

    :param ndarray pred: model prediction.
    :param ndarray gt: ground truth.
    :param ndarray mask: used to maskout on-sea values. We focus values on land. Default mask location is at "../mask/mask_sd5km.npy".
    :param float threshold: is set according to the levels of raining amplitudes.
    :return int: number of counts that both predicted pixel values and ground truth pixel values are 1 (True).
    '''
    gt_false = (gt<threshold).astype(int)
    hit = (pred>=threshold).astype(int)
    hit = hit*gt_false*mask
    return hit[hit>0].shape[0]

def CorrectNegative(pred, gt, mask, threshold)->int:
    '''
    Binarize predicted pixel values and ground truth pixel values with the thresholds.
    A "Correct Negative" is counted when "event forecast NOT to occur, but did NOT occur".

    :param ndarray pred: model prediction.
    :param ndarray gt: ground truth.
    :param ndarray mask: used to maskout on-sea values. We focus values on land. Default mask location is at "../mask/mask_sd5km.npy".
    :param float threshold: is set according to the levels of raining amplitudes.
    :return int: number of counts that both predicted pixel values and ground truth pixel values are 1 (True).
    '''
    gt_false = (gt<threshold).astype(int)
    miss = (pred<threshold).astype(int)
    miss = miss*gt_false*mask
    return miss[miss>0].shape[0]

def Accuracy(pred, gt, mask, threshold)->float:
    assert np.size(pred) == np.size(gt) == np.size(mask)
    total = np.size(pred)
    return (Hit(pred, gt, mask, threshold) + CorrectNegative(pred, gt, mask, threshold)) / total