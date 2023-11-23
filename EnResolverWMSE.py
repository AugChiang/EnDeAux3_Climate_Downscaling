from Encoder_uv.model import Encoder, Resovler
from Encoder_uv.dataloader import MyDataset, datagen, Scale01, ScaleNeg11
import numpy as np
import os
from glob import glob
import tensorflow as tf
import random
import math
import scipy.stats
from tensorflow.image import ssim
from tensorflow import get_static_value
import pandas as pd

# Define constants
num_epochs = 1000
checkpt = 1
batch_size = 64
xn, xm = 14, 9
vector_length = 126
scale = 5
random_seed = None
split = 0.9
target_sequence_length = vector_length  # Target sequence length matches input
x_max = 31.544506
y_max = 1401.9225

# Initialize the model with a one-unit output layer
num_layers = 4
d_model = 128 # Dimension of embeddings
num_heads = 6  # Number of attention heads
dff = 128

# Dimension of feedforward layer
x_train_path = "ERA5_tp_14x9"
y_train_path = "sd0_5km"

mask = np.load("mask/mask_sd5km.npy")
mask = np.reshape(mask, (xn*scale, xm*scale, 1))
aux_path = {'lr':'sd_25km',
            'u':'u_npy', 'v':'v_npy'}
channel = 1 + len(aux_path)
save_dir = "EnRe_AUX3_dff128_WMSE"

# early stopping constants
patience = 20
wait = 0
best = float('inf')

wmse_gamma = 1.0
class wmse(tf.keras.losses.Loss):
    def __init__(self, wmse_gamma = wmse_gamma):
        super().__init__()
        self.gamma = wmse_gamma

    def call(self, y_true, y_pred):
        if y_true.shape[1]!=y_pred.shape[1] or y_true.shape[2]!=y_pred.shape[2]:
            y = tf.image.resize(y_true, [y_pred.shape[1],y_pred.shape[2]], method='bilinear')
        else:
            y = y_true
        se = tf.math.square(y_pred-y)
        wse = (1-self.gamma)*se + self.gamma*tf.math.multiply(se, y)
        wmse = tf.reduce_mean(wse)
        return wmse


# Define loss function (e.g., mean squared error) and optimizer (e.g., Adam)
# loss_object = tf.keras.losses.MeanSquaredError()
# loss_object = tf.keras.losses.KLDivergence()
loss_object = wmse()
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)  # encoder's
optimizer2 = tf.keras.optimizers.Adam(learning_rate=5e-5) # resolver's

# np.log1p and [0,1], [-1,1] for aux data
dataset = MyDataset(xtrpath=x_train_path, ytrpath=y_train_path, aux_path=aux_path,
                    shuffle_size=batch_size*4, batch_size=batch_size, seed=random_seed,
                    x_max=x_max, y_max=y_max, size=((vector_length, channel),(xn*scale, xm*scale,1)))

myresolver = Resovler(scale=scale)
myencoder = Encoder(xn = xn, xm = xm,
                    num_layers=num_layers, d_model=d_model,
                    num_heads=num_heads, dff=dff,
                    input_vocab_size=vector_length,  # Input vocabulary size matches vector_length
                    channel=channel, dropout_rate=0.2)
# compile the model
encoder_input = tf.keras.layers.Input(shape=(vector_length, channel))
encoder_output = myencoder(encoder_input, training=True, mask=None)
resolver_input = tf.keras.layers.Input(shape=(xn, xm, 1))
resolver_output = myresolver(resolver_input, training=True)

# Create the model
model_encoder = tf.keras.models.Model(inputs=encoder_input, outputs=encoder_output)
model_resolver = tf.keras.models.Model(inputs=resolver_input, outputs=resolver_output)
model_encoder.compile(optimizer=[optimizer], loss=loss_object)
model_resolver.compile(optimizer=[optimizer2], loss=loss_object)
# model.summary()

# Training step
# @tf.function
# def train_step(inputs, labels, model, lr=False):
#     with tf.GradientTape() as tape:
#         predictions = model(inputs, mask=None)
#         loss = loss_object(labels, predictions, lr=lr)

#     gradients = tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#     return loss

# @tf.function
# def val_step(inputs, labels, model, lr=False):
#     with tf.GradientTape() as tape:
#         predictions = model(inputs, mask=None)
#         loss = loss_object(labels, predictions, lr=lr)

#     return loss
def train(save_dir=save_dir, num_epochs=num_epochs, patience=patience, wait=wait, best=best):
    tr_history = []
    val_history = []
    # dataset
    train_batch = dataset.train_dataset_gen()
    val_batch = dataset.val_dataset_gen()
    # Training loop
    # training the encoder, freeze the resolver
    for epoch in range(num_epochs):
        tr_loss = []
        val_loss = []

        for step, (input_data, labels) in enumerate(train_batch):
            with tf.GradientTape() as tape:
                predictions = myencoder(input_data, training=True, mask=None)
                loss = loss_object(labels, predictions)

            gradients = tape.gradient(loss, myencoder.trainable_variables)
            optimizer.apply_gradients(zip(gradients, myencoder.trainable_variables))
            tr_loss.append(loss)

        for step, (val_data, val_labels) in enumerate(val_batch):
            predictions = myencoder(val_data,  training=False, mask=None)
            loss = loss_object(val_labels, predictions)
            val_loss.append(loss)

        avg_tr_loss = sum(tr_loss) / len(tr_loss)
        avg_val_loss = sum(val_loss) / len(val_loss)

        tr_history.append(avg_tr_loss)
        val_history.append(avg_val_loss)

        print(f"Epoch {epoch + 1}, Loss: {avg_tr_loss.numpy()}, Val_loss: {avg_val_loss.numpy()}")
        wait += 1

        # if num_epochs % checkpt == 0:
        #      checkpoint.save(os.path.join(save_dir, 'training_checkpoints'))

        if(avg_val_loss < best):
            best = avg_val_loss
            wait = 0
        if(wait >= patience):
            break

    # save the trained model
    if not os.path.exists(save_dir):
        os.mkdir(save_dir) 

    model_encoder.save_weights(save_dir + f"/encoder_variables")
    print("Complete Saving the Model Weights...")
    np.save(os.path.join(save_dir,  'encoder_losses.npy'), np.array(tr_history))
    np.save(os.path.join(save_dir, 'encoder_val_losses.npy'), np.array(val_history))
    print("Complete Saving the Losses...")

    # reset early stopping constants
    patience = 20
    wait = 0
    best = float('inf')

    tr_history.clear()
    val_history.clear()

    # training the resolver, freeze the encoder
    for epoch in range(num_epochs):
        tr_loss = []
        val_loss = []

        for step, (input_data, labels) in enumerate(train_batch):
            tr_encode = model_encoder.predict(input_data)
            with tf.GradientTape() as tape:
                predictions = model_resolver(tr_encode, training=True)
                loss = loss_object(labels, predictions)

            gradients = tape.gradient(loss, myresolver.trainable_variables)
            optimizer2.apply_gradients(zip(gradients, myresolver.trainable_variables))
            tr_loss.append(loss)

        for step, (val_data, val_labels) in enumerate(val_batch):
            val_encode = model_encoder.predict(val_data)
            with tf.GradientTape() as tape:
                predictions = model_resolver(val_encode, training=False)
                loss = loss_object(val_labels, predictions)
            val_loss.append(loss)

        avg_tr_loss = sum(tr_loss) / len(tr_loss)
        avg_val_loss = sum(val_loss) / len(val_loss)

        tr_history.append(avg_tr_loss)
        val_history.append(avg_val_loss)

        print(f"Epoch {epoch + 1}, Loss: {avg_tr_loss.numpy()}, Val_loss: {avg_val_loss.numpy()}")
        wait += 1

        # if num_epochs % checkpt == 0:
        #      checkpoint.save(os.path.join(save_dir, 'training_checkpoints'))

        if(avg_val_loss < best):
            best = avg_val_loss
            wait = 0
        if(wait >= patience):
            break

    model_resolver.save_weights(save_dir + f"/resolver_variables")
    print("Complete Saving the Model Weights...")
    np.save(os.path.join(save_dir,  'resolver_losses.npy'), np.array(tr_history))
    np.save(os.path.join(save_dir, 'resolver_val_losses.npy'), np.array(val_history))
    print("Complete Saving the Losses...")

def getdatapath(path_list, split=split, seed=random_seed):
    if(seed is not None):
        random.Random(seed).shuffle(path_list) # shuffle the paths with seed
    train_len = math.floor(len(path_list)*split)
    test_len = math.floor((len(path_list) - train_len)/2)
    # train, val, test
    return path_list[:train_len], path_list[train_len:train_len+test_len], path_list[train_len+test_len:]


def res_eval(x, date, mask, num_valid_grid_points):
    gt = np.load(os.path.join("sd0_5km", date)).astype('float32')
    gt = np.reshape(gt, (70,45))
    # x = x.astype('float32')
    x = np.reshape(x, (70,45))

    diff = abs((gt - x)*mask) # pixel-wise difference

    # mean absolute error
    mae = np.sum(diff)/num_valid_grid_points

    # root mean square error
    rmse = np.sqrt(np.sum(np.square(diff))/num_valid_grid_points)

    # Pearson Correlation
    flat_gt = gt.flatten()
    flat_res = x.flatten()
    corr = scipy.stats.pearsonr(flat_res, flat_gt)[0] # drop p_values

    # SSIM
    gt_max = np.max(gt)
    ssim_value = ssim(np.expand_dims(x, axis=-1),
                      np.expand_dims(gt, axis=-1),
                      max_val=gt_max)
    ssim_value = get_static_value(ssim_value)

    return mae, rmse, corr, ssim_value

def pred():
    val_pred_save_dir = os.path.join(save_dir, 'val_pred')
    test_pred_save_dir = os.path.join(save_dir, 'test_pred')

    mask = np.load("mask/mask_sd5km.npy")
    mask = np.reshape(mask, (70,45))
    num_valid_grid_points = mask[mask>0].shape[0]

    if not os.path.exists(val_pred_save_dir):
        os.mkdir(val_pred_save_dir)
    if not os.path.exists(test_pred_save_dir):
        os.mkdir(test_pred_save_dir)

    model_encoder.load_weights(save_dir + f"/encoder_variables")
    model_resolver.load_weights(save_dir + f"/resolver_variables")
    data_paths = glob(x_train_path + '/*.npy')
    data_paths.sort()
    # print("len of data paths: ", len(data_paths))
    _, val_paths, test_paths = getdatapath(data_paths)
    val_data = datagen(val_paths, aux_path=aux_path, x_max=x_max, x_size=(1, vector_length, channel))
    test_data = datagen(test_paths, aux_path=aux_path, x_max=x_max, x_size=(1, vector_length, channel))

    # dataframe
    df = {'Date':[], 'MAE':[], 'RMSE':[], 'Corr':[], 'SSIM':[]}
    for i, (x, date) in enumerate(val_data):
        pred = model_encoder.predict(x)
        pred = model_resolver.predict(pred)

        # denormalize and flatten
        pred = np.expm1(pred*np.log1p(y_max)).flatten()

        # save prediction
        np.save(os.path.join(val_pred_save_dir, date), pred)

        # eval metrics
        mae, rmse, corr, ssim_value = res_eval(pred, date, mask, num_valid_grid_points)
        df['Date'].append(date)
        df['MAE'].append(mae)
        df['RMSE'].append(rmse)
        df['Corr'].append(corr)
        df['SSIM'].append(ssim_value)

        print(f"Exporting ... {date}")

    # save file
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(save_dir, 'val_eval.csv'))
    with open(os.path.join(save_dir, 'val_eval.txt'), 'a') as f:
        f.write("Val, MAE, RMSE, Corr, SSIM\n")
        f.write(f"AVG, {str(df['MAE'].mean())}, {str(df['RMSE'].mean())}, {str(df['Corr'].mean())}, {str(df['SSIM'].mean())}\n")
        f.write(f"MED, {str(df['MAE'].median())}, {str(df['RMSE'].median())}, {str(df['Corr'].median())}, {str(df['SSIM'].median())}")

    # dataframe
    df = {'Date':[], 'MAE':[], 'RMSE':[], 'Corr':[], 'SSIM':[]}
    for i, (x, date) in enumerate(test_data):
        pred = model_encoder.predict(x)
        pred = model_resolver.predict(pred)

        # denormalize and flatten
        pred = np.expm1(pred*np.log1p(y_max)).flatten()

        # eval metrics
        mae, rmse, corr, ssim_value = res_eval(pred, date, mask, num_valid_grid_points)
        df['Date'].append(date)
        df['MAE'].append(mae)
        df['RMSE'].append(rmse)
        df['Corr'].append(corr)
        df['SSIM'].append(ssim_value)

        # save prediction
        np.save(os.path.join(test_pred_save_dir, date), pred)
        print(f"Exporting ... {date}")
    
    # save file
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(save_dir, 'test_eval.csv'))
    with open(os.path.join(save_dir, 'test_eval.txt'), 'a') as f:
        f.write("Val, MAE, RMSE, Corr, SSIM\n")
        f.write(f"AVG, {str(df['MAE'].mean())}, {str(df['RMSE'].mean())}, {str(df['Corr'].mean())}, {str(df['SSIM'].mean())}\n")
        f.write(f"MED, {str(df['MAE'].median())}, {str(df['RMSE'].median())}, {str(df['Corr'].median())}, {str(df['SSIM'].median())}")

def npytotxt():
    mask = np.load("mask/mask_sd5km.npy")
    mask = np.reshape(mask, (70,45))
    latx5 = np.linspace(start=25.25, stop=22., num=xn*5, endpoint=True)
    lonx5 = np.linspace(start=120., stop=122., num=xm*5, endpoint=True)
    val_pred_txt_save_dir = os.path.join(save_dir, 'val_pred_txt')
    test_pred_txt_save_dir = os.path.join(save_dir, 'test_pred_txt')
    val_pred_paths = glob(os.path.join(save_dir, 'val_pred', '*.npy'))
    test_pred_paths = glob(os.path.join(save_dir, 'test_pred', '*.npy'))

    if not os.path.exists(val_pred_txt_save_dir):
        os.mkdir(val_pred_txt_save_dir)
    if not os.path.exists(test_pred_txt_save_dir):
        os.mkdir(test_pred_txt_save_dir)

    def writetxt(paths, saveto):
        for file in paths:
            date = file[-12:-4]
            pred = np.load(file)
            pred = np.reshape(pred, (70,45))
            with open(os.path.join(saveto, f"{date}.txt"), 'a') as f:
                f.write("lat, lon, precipitation(mm) \n")
                for row in range(70):
                    for col in range(45):
                        if mask[row][col]:
                            f.write(str(latx5[row]) + ',') # lat
                            f.write(str(lonx5[col]) + ',') # lon
                            f.write(str(pred[row][col]) + '\n') # precipitation value (mm)
            print(f"Exporting {date}...")

    writetxt(val_pred_paths, val_pred_txt_save_dir)
    writetxt(test_pred_paths, test_pred_txt_save_dir)
    print("Convert npy to txt completed.")

    return 0

def QoF():
    H = []
    M = []
    FA = []
    CN = []
    DATE = []
    threshold = [80, 200, 350, 500]
    mask = np.load("mask/mask_sd5km.npy")
    mask = np.reshape(mask, (70,45))

    qof_save_dir = os.path.join(save_dir, 'QoFs')
    if not os.path.exists(qof_save_dir):
        os.mkdir(qof_save_dir)

    val_pred_paths = glob(os.path.join(save_dir, 'val_pred', '*.npy'))
    test_pred_paths = glob(os.path.join(save_dir, 'test_pred', '*.npy'))

    def Hit(pred, gt, mask, threshold)->int:
        gt_hit = (gt>=threshold).astype(int)
        hit = (pred>=threshold).astype(int)
        hit = hit*gt_hit*mask
        return hit[hit>0].shape[0]

    def Miss(pred, gt, mask, threshold)->int:
        gt_hit = (gt>=threshold).astype(int)
        miss = (pred<threshold).astype(int)
        miss = miss*gt_hit*mask
        return miss[miss>0].shape[0]

    def FalseAlarm(pred, gt, mask, threshold)->int:
        gt_false = (gt<threshold).astype(int)
        hit = (pred>=threshold).astype(int)
        hit = hit*gt_false*mask
        return hit[hit>0].shape[0]

    def CorrectNegative(pred, gt, mask, threshold)->int:
        gt_false = (gt<threshold).astype(int)
        miss = (pred<threshold).astype(int)
        miss = miss*gt_false*mask
        return miss[miss>0].shape[0]

    def writeqoftxt(paths, datatype:str, saveto=qof_save_dir):
        flat_mask = mask.flatten()
        for thre in threshold:
            name = datatype + '_' + str(thre) + 'mm.txt' # e.g. val_80mm.txt
            with open(os.path.join(saveto, name), 'a') as f:
                f.write("Date, Hit, Miss, FA, CN")
                f.write('\n')
                for n, path in enumerate(paths):
                    pred = np.load(path).flatten()
                    date = path[-12:] # yyyymmdd.npy
                    gt = np.load(os.path.join(y_train_path, date)).flatten()

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
    print("Task Completed.")

    return 0

if __name__ == '__main__':
    train()
    pred()
    npytotxt()
    QoF()
