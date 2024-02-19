from Encoder_uv.model import Encoder, Resovler
from Encoder_uv.dataloader import MyDataset, datagen, Scale01, ScaleNeg11, GetTopology
from Encoder_uv import loss
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
import configparser

# 
config = configparser.ConfigParser()
config.read('config.ini')

# resolution related
xn = config['resolution'].getint('input_height')
xm = config['resolution'].getint('input_width')
topo_x = config['resolution'].getint('topo_height')
topo_y = config['resolution'].getint('topo_width')
vector_length = config['resolution'].getint('vector_length')
target_sequence_length = vector_length  # Target sequence length matches input

# scsaling factor
scale = config['scale_factor'].getint('scale')

# model training var
num_epochs = config['training'].getint('num_epochs')
checkpt = config['training'].getint('check_pt')
batch_size = config['training'].getint('batch_size')
encoder_lr = config['training'].getfloat('encoder_lr')
resolver_lr = config['training'].getfloat('resolver_lr')

# data
try:
    random_seed = config['data'].getint('random_seed')
except ValueError:
    random_seed = None

split = config['data'].getfloat('split')
x_max = config['data'].getfloat('input_total_max')
y_max = config['data'].getfloat('label_total_max')
shuffle_fac = config['data'].getint('shuffle_factor') # n times of batch_size as shuffle buffer size
# sampling_a = config['data'].getfloat('sampling_fac') # sampling strategy, see 'dataloader.py'

# model config
num_layers = config['model'].getint('num_layers')
d_model = config['model'].getint('embedding_size') # Dimension of embeddings
num_heads = config['model'].getint('num_attention_heads')  # Number of attention heads
dff = config['model'].getint('feed_forward_size') # feed forward projection size
drop_rate = config['model'].getfloat('drop_rate')

# normalization option
x_use_log1 = config['normalization'].getboolean('input_use_log1') # log(x+1)
y_use_log1 = config['normalization'].getboolean('label_use_log1') # log(x+1)
norm_01 = config['normalization'].getboolean('norm_01')
topo_use_log1 = config['normalization'].getboolean('topo_use_log1')
topo_norm_01 = config['normalization'].getboolean('topo_norm_01')

# path
x_train_path = config['path']['x_training_path']
y_train_path = config['path']['y_training_path']
topo_path = config['path']['topo_path']
mask_path = config['path']['mask_path']
save_dir = config['path']['save_path']

mask = np.load(mask_path)
mask = np.reshape(mask, (xn*scale, xm*scale, 1))
aux_path = dict(config['aux'])
channel = 1 + len(aux_path)


# early stopping
encoder_patience = config['early_stop'].getint('encoder_patience')
resolver_patience = config['early_stop'].getint('resolver_patience')
wait = 0
best = float('inf')

# weighted loss function
wmse_gamma = config['loss_gamma'].getfloat('wmse_gamma')


# Define loss function (e.g., mean squared error)
# and optimizer (e.g., Adam)
loss_object = loss.wmse()
optimizer = tf.keras.optimizers.Adam(learning_rate=encoder_lr)  # encoder's
optimizer2 = tf.keras.optimizers.Adam(learning_rate=resolver_lr) # resolver's

# np.log1p and [0,1], [-1,1] for aux data
dataset = MyDataset(xtrpath=x_train_path, ytrpath=y_train_path, aux_path=aux_path,
                    x_use_log1=x_use_log1, y_use_log1=y_use_log1,
                    shuffle_size=batch_size*shuffle_fac, batch_size=batch_size, seed=random_seed,
                    x_max=x_max, y_max=y_max, size=((vector_length, channel),(xn*scale, xm*scale,1)))
# topography
topo = GetTopology(topo_path=topo_path,
                   topo_x=topo_x, topo_y=topo_y,
                   y_n=xn*scale, y_m=xm*scale,
                   use_log1=topo_use_log1, use_01=topo_norm_01)

myresolver = Resovler(scale=scale, topo=topo)
myencoder = Encoder(xn = xn, xm = xm,
                    num_layers=num_layers, d_model=d_model,
                    num_heads=num_heads, dff=dff,
                    input_vocab_size=vector_length,  # Input vocabulary size matches vector_length
                    channel=channel, dropout_rate=drop_rate)
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
def train(save_dir=save_dir, num_epochs=num_epochs):
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
        if(wait >= encoder_patience):
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
        if(wait >= resolver_patience):
            break

    model_resolver.save_weights(save_dir + f"/resolver_variables")
    print("Complete Saving the Model Weights...")
    np.save(os.path.join(save_dir,  'resolver_tr_losses.npy'), np.array(tr_history))
    np.save(os.path.join(save_dir, 'resolver_val_losses.npy'), np.array(val_history))
    print("Complete Saving the Losses...")

def getdatapath(path_list, split=split, seed=random_seed):
    if(seed is not None):
        random.Random(seed).shuffle(path_list) # shuffle the paths with seed
    train_len = math.floor(len(path_list)*split)
    test_len = math.floor((len(path_list) - train_len)/2)
    # train, val, test
    return path_list[:train_len], path_list[train_len:train_len+test_len], path_list[train_len+test_len:]

# evaluation of x,y
def res_eval(x, date, mask, num_valid_grid_points, y_path=y_train_path):
    gt = np.load(os.path.join(y_path, date)).astype('float32')
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

def pred(mask=mask):
    val_pred_save_dir = os.path.join(save_dir, 'val_pred')
    test_pred_save_dir = os.path.join(save_dir, 'test_pred')

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
    df.to_csv(os.path.join(save_dir, 'val_eval_topo.csv'))
    with open(os.path.join(save_dir, 'val_eval_topo.txt'), 'a') as f:
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
        f.write("Test, MAE, RMSE, Corr, SSIM\n")
        f.write(f"AVG, {str(df['MAE'].mean())}, {str(df['RMSE'].mean())}, {str(df['Corr'].mean())}, {str(df['SSIM'].mean())}\n")
        f.write(f"MED, {str(df['MAE'].median())}, {str(df['RMSE'].median())}, {str(df['Corr'].median())}, {str(df['SSIM'].median())}")

# save as .txt file
def npytotxt(mask=mask):
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

# forecast indicators
def QoF(mask=mask):
    H = []
    M = []
    FA = []
    CN = []
    DATE = []
    threshold = [80, 200, 350, 500]
    mask = np.reshape(mask, (70,45))

    qof_save_dir = os.path.join(save_dir, 'QoFs_topo')
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
