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

    Parameters
    -----
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
    '''
    Fraction correct, the fraction of the forecasts that are correct.
    range: [0,1], perfect score = 1

    Parameters
    -----
    :param ndarray pred: model prediction.
    :param ndarray gt: ground truth.
    :param ndarray mask: used to maskout on-sea values. We focus values on land. Default mask location is at "../mask/mask_sd5km.npy".
    :param float threshold: is set according to the levels of raining amplitudes.
    :return float: _description_
    '''
    assert np.size(pred) == np.size(gt) == np.size(mask)
    total = np.size(pred)
    return (Hit(pred, gt, mask, threshold) +
            CorrectNegative(pred, gt, mask, threshold)) / total

def BiasScore(pred, gt, mask, threshold)->float:
    '''
    Frequenct bias, showing the forecast frequenct of "True" events compare to the observed frequency of "True" events.
    range: [0, inf), perfect score = 1

    Parameters
    -----
    :param ndarray pred: model prediction.
    :param ndarray gt: ground truth.
    :param ndarray mask: used to maskout on-sea values. We focus values on land. Default mask location is at "../mask/mask_sd5km.npy".
    :param float threshold: is set according to the levels of raining amplitudes.
    :return float: _description_
    '''
    assert np.size(pred) == np.size(gt) == np.size(mask)
    return (Hit(pred, gt, mask, threshold) +
            FalseAlarm(pred, gt, mask, threshold)) / (Hit(pred, gt, mask, threshold) + Miss(pred, gt, mask, threshold))

def PoD(pred, gt, mask, threshold)->float:
    '''
    Probability of Detection, i.e., the hit rate, fraction of the observed "True" events that are correctly forecast.
    range: [0,1], perfect score = 1

    Parameters
    -----
    :param ndarray pred: model prediction.
    :param ndarray gt: ground truth.
    :param ndarray mask: used to maskout on-sea values. We focus values on land. Default mask location is at "../mask/mask_sd5km.npy".
    :param float threshold: is set according to the levels of raining amplitudes.
    :return float: _description_
    '''
    assert np.size(pred) == np.size(gt) == np.size(mask)
    return Hit(pred, gt, mask, threshold) / (Hit(pred, gt, mask, threshold) + Miss(pred, gt, mask, threshold))

def FAR(pred, gt, mask, threshold)->float:
    '''
    False Alarm Ratio, the fraction of the predicted "True" events actually did NOT occur (i.e. were False Alarm).
    range: [0,1], perfect score = 0

    Parameters
    -----
    :param ndarray pred: model prediction.
    :param ndarray gt: ground truth.
    :param ndarray mask: used to maskout on-sea values. We focus values on land. Default mask location is at "../mask/mask_sd5km.npy".
    :param float threshold: is set according to the levels of raining amplitudes.
    :return float: _description_
    '''
    assert np.size(pred) == np.size(gt) == np.size(mask)
    return FalseAlarm(pred, gt, mask, threshold) / (Hit(pred, gt, mask, threshold) + FalseAlarm(pred, gt, mask, threshold))

def PoFD(pred, gt, mask, threshold)->float:
    '''
    False Alarm Rate, the fraction of the observed "False" events that were incorrectly forecast as "True".
    range: [0,1], perfect score = 0

    Parameters
    -----
    :param ndarray pred: model prediction.
    :param ndarray gt: ground truth.
    :param ndarray mask: used to maskout on-sea values. We focus values on land. Default mask location is at "../mask/mask_sd5km.npy".
    :param float threshold: is set according to the levels of raining amplitudes.
    :return float: _description_
    '''
    assert np.size(pred) == np.size(gt) == np.size(mask)
    return FalseAlarm(pred, gt, mask, threshold) / (CorrectNegative(pred, gt, mask, threshold) + FalseAlarm(pred, gt, mask, threshold))

def TS(pred, gt, mask, threshold)->float:
    '''
    Threat Score (critical success index), showing how well the forecast "True" events correspond to the observed "True" events.
    range: [0,1], where 0 indicates no skill. perfect score = 1

    Parameters
    -----
    :param ndarray pred: model prediction.
    :param ndarray gt: ground truth.
    :param ndarray mask: used to maskout on-sea values. We focus values on land. Default mask location is at "../mask/mask_sd5km.npy".
    :param float threshold: is set according to the levels of raining amplitudes.
    :return float: _description_
    '''
    assert np.size(pred) == np.size(gt) == np.size(mask)
    return Hit(pred, gt, mask, threshold) / (Hit(pred, gt, mask, threshold) + Miss(pred, gt, mask, threshold) + FalseAlarm(pred, gt, mask, threshold))
