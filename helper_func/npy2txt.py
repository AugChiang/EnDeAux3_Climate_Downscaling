import os
import numpy as np
from glob import glob

# save as .txt file
def npytotxt(yn:int, ym:int, lat, lon, mask, root_dir:str, saveto:str)->None:
    '''
    Load given numpy arrays within the given root dir and write to text file.
    Each row represents a precipitation value and its location (latitude, longitude).

    Parameters
    -----
    :param int yn: height of the data in root_dir
    :param int ym: widht of the data in root_dir
    :param array_like lat: array of latitude points.
    :param array_like lon: array of longitude points.
    :param _type_ mask: to maskout on-sea values
    :param str root_dir: root dir of the numpy files
    :param str saveto: path for saving text files.
    -----
    :return None
    '''
    assert len(lat) == yn
    assert len(lon) == ym
    if len(mask.shape) == 1:
        mask = np.reshape(mask, (yn, ym))
    
    paths = glob(os.path.join(root_dir, '*.npy'))
    if not os.path.exists(saveto):
        os.mkdir(saveto)

    for file in paths:
        date = file[-12:-4] # default: yyyymmdd
        pred = np.load(file)
        if len(pred.shape) != 2:
            pred = np.reshape(pred, (yn,ym))

        with open(os.path.join(saveto, f"{date}.txt"), 'a') as f:
            f.write("lat, lon, precipitation(mm) \n")
            for row in range(yn):
                for col in range(ym):
                    if mask[row][col]:
                        f.write(str(lat[row]) + ',') # lat
                        f.write(str(lon[col]) + ',') # lon
                        f.write(str(pred[row][col]) + '\n') # precipitation value (mm)
        print(f"Exporting {date}.txt...")

    print("Convert npy to txt completed.")