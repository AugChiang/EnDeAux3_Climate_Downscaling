from Encoder_uv.model import Encoder, Resovler
from Encoder_uv.dataloader import MyDataset, GetTopology
from Encoder_uv import loss
from helper_func.data_gen import datagen, getdatapath
from helper_func.eval import res_eval
from helper_func.npy2txt import npytotxt
from ForecastIndicators.qof import QoF # forecast indicators
import numpy as np
import os
from glob import glob
import tensorflow as tf
import random
import math
import pandas as pd
import configparser

# 
config = configparser.ConfigParser()
config.read('config.ini')

# resolution related
xn = config['resolution'].getint('input_height')
xm = config['resolution'].getint('input_width')
x_ch = config['resolution'].getint('input_ch')
topo_x = config['resolution'].getint('topo_height')
topo_y = config['resolution'].getint('topo_width')
vector_length = config['resolution'].getint('vector_length')
target_sequence_length = vector_length  # Target sequence length matches input

# scaling factor
scale = config['scale_factor'].getint('scale')

# final output (default: precipitation only)
yn = xn*scale # label height
ym = xm*scale # label width
y_ch = config['resolution'].getint('output_ch') # label channel

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
# sampling_fac = config['data'].getfloat('sampling_fac') # sampling strategy, see 'dataloader.py'

# model config
num_layers = config['model'].getint('num_layers')
d_model = config['model'].getint('embedding_size') # Dimension of embeddings
num_heads = config['model'].getint('num_attention_heads')  # Number of attention heads
dff = config['model'].getint('feed_forward_size') # feed forward projection size
drop_rate = config['model'].getfloat('drop_rate')

# normalization option
x_use_log1 = config['normalization'].getboolean('input_use_log1') # log(x+1)
y_use_log1 = config['normalization'].getboolean('label_use_log1') # log(x+1)
norm_01 = config['normalization'].getboolean('norm_01') # linearly normalization to [0,1]
topo_use_log1 = config['normalization'].getboolean('topo_use_log1')
topo_norm_01 = config['normalization'].getboolean('topo_norm_01')

# path
x_train_path = config['path']['x_training_path']
y_train_path = config['path']['y_training_path']
topo_path = config['path']['topo_path']
mask_path = config['path']['mask_path']
save_dir = config['path']['save_path']
# root dir of all results
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# encoder and resolver weights save dir
weights_save_dir = os.path.join(save_dir, "weights")
if not os.path.exists(weights_save_dir):
    os.mkdir(weights_save_dir)

mask = np.load(mask_path)
mask = np.reshape(mask, (yn, ym, 1))
aux_path = dict(config['aux'])
channel = x_ch + (len(aux_path)-1) # number of aux (except "stat" data) + the number of channel of training data


# early stopping
encoder_patience = config['early_stop'].getint('encoder_patience')
resolver_patience = config['early_stop'].getint('resolver_patience')


# weighted loss function
wmse_gamma = config['loss_gamma'].getfloat('wmse_gamma')


# Define loss function (e.g., mean squared error)
# and optimizer (e.g., Adam)
loss_object = loss.wmse(wmse_gamma)
optimizer = tf.keras.optimizers.Adam(learning_rate=encoder_lr)  # encoder's
optimizer2 = tf.keras.optimizers.Adam(learning_rate=resolver_lr) # resolver's

# np.log1p and [0,1], [-1,1] for aux data
dataset = MyDataset(xtrpath=x_train_path,
                    ytrpath=y_train_path,
                    aux_path=aux_path,
                    x_use_log1=x_use_log1,
                    y_use_log1=y_use_log1,
                    shuffle_size=batch_size*shuffle_fac,
                    batch_size=batch_size,
                    seed=random_seed,
                    x_max=x_max,
                    y_max=y_max,
                    size=((vector_length, channel),(yn, ym, y_ch)),
                    split=split)
# topography
topo = GetTopology(topo_path=topo_path,
                   topo_x=topo_x,
                   topo_y=topo_y,
                   y_n=yn,
                   y_m=ym,
                   use_log1=topo_use_log1,
                   use_01=topo_norm_01)

myresolver = Resovler(scale=scale,
                      topo=topo,
                      y_ch=y_ch)

myencoder = Encoder(xn = xn,
                    xm = xm,
                    num_layers=num_layers,
                    d_model=d_model,
                    num_heads=num_heads,
                    dff=dff,
                    input_vocab_size=vector_length,  # Input vocabulary size matches vector_length
                    channel=channel,
                    dropout_rate=drop_rate)
# compile the model
encoder_input = tf.keras.layers.Input(shape=(vector_length, channel))
encoder_output = myencoder(encoder_input, training=True, mask=None)
resolver_input = tf.keras.layers.Input(shape=(xn, xm, x_ch)) # resolver output default ch = 1
resolver_output = myresolver(resolver_input, training=True)

# Create the model
model_encoder = tf.keras.models.Model(inputs=encoder_input, outputs=encoder_output)
model_resolver = tf.keras.models.Model(inputs=resolver_input, outputs=resolver_output)
model_encoder.compile(optimizer=[optimizer], loss=loss_object)
model_resolver.compile(optimizer=[optimizer2], loss=loss_object)
# model_encoder.summary()
# model_resolver.summary()

# Training loop
def train(save_dir=save_dir, num_epochs=num_epochs):
    '''
    Training loops.
    The 1st for-loop over epochs is to train the encoder.
    The 2nd one is to train the resolver while freezing the encoder weights.

    :param str save_dir: destination of saving the results, including model weights, predictions, evals, etc, defaults to save_dir
    :param int num_epochs: defaults to num_epochs
    '''
    # losses saving
    loss_save_dir = os.path.join(save_dir, "losses")
    if not os.path.exists(loss_save_dir):
        os.mkdir(loss_save_dir)
    tr_history = []
    val_history = []
    # dataset
    train_batch = dataset.train_dataset_gen()
    val_batch = dataset.val_dataset_gen()

    # init of early stopping
    wait = 0
    best = float('inf')

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

    # save model weights
    model_encoder.save_weights(os.path.join(weights_save_dir, "encoder_variables"))
    print("Complete Saving the Encoder Weights...")

    # save losses as npy files
    np.save(os.path.join(loss_save_dir,  'encoder_losses.npy'), np.array(tr_history))
    np.save(os.path.join(loss_save_dir, 'encoder_val_losses.npy'), np.array(val_history))
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

    # save resolver weights
    model_resolver.save_weights(os.path.join(weights_save_dir, "resolver_variables"))
    print("Complete Saving the Model Weights...")

    # save losses as npy
    np.save(os.path.join(loss_save_dir,  'resolver_tr_losses.npy'), np.array(tr_history))
    np.save(os.path.join(loss_save_dir, 'resolver_val_losses.npy'), np.array(val_history))
    print("Complete Saving the Losses...")

def pred(mask):
    val_pred_save_dir = os.path.join(save_dir, 'val_pred')
    test_pred_save_dir = os.path.join(save_dir, 'test_pred')


    if not os.path.exists(val_pred_save_dir):
        os.mkdir(val_pred_save_dir)
    if not os.path.exists(test_pred_save_dir):
        os.mkdir(test_pred_save_dir)

    model_encoder.load_weights(os.path.join(weights_save_dir, "encoder_variables"))
    model_resolver.load_weights(os.path.join(weights_save_dir, "resolver_variables"))
    data_paths = glob(os.path.join(x_train_path, '*.npy'))
    data_paths.sort()
    # print("len of data paths: ", len(data_paths))
    _, val_paths, test_paths = getdatapath(data_paths, split=split, seed=random_seed)
    val_data = datagen(val_paths, aux_path=aux_path, x_max=x_max, x_size=(1, vector_length, channel))
    test_data = datagen(test_paths, aux_path=aux_path, x_max=x_max, x_size=(1, vector_length, channel))

    # dataframe
    df = {'Date':[], 'MAE':[], 'RMSE':[], 'Corr':[], 'SSIM':[]}
    for i, (x, date) in enumerate(val_data):
        file_name = date + '.npy'
        pred = model_encoder.predict(x)
        pred = model_resolver.predict(pred)

        # denormalize and flatten
        pred = np.expm1(pred*np.log1p(y_max)).flatten()

        # save prediction
        np.save(os.path.join(val_pred_save_dir, file_name), pred)

        # load corresponding label data (ground truth data)
        gt = np.load(os.path.join(y_train_path, file_name)).astype('float32')
        gt = np.reshape(gt, (yn,ym))
        pred = np.reshape(pred, (yn,ym))

        # eval metrics
        mae, rmse, corr, ssim_value = res_eval(pred, gt, mask)
        df['Date'].append(date)
        df['MAE'].append(mae)
        df['RMSE'].append(rmse)
        df['Corr'].append(corr)
        df['SSIM'].append(ssim_value)

        print(f"Exporting pred on {date}...")

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
        file_name = date + '.npy'
        pred = model_encoder.predict(x)
        pred = model_resolver.predict(pred)

        # denormalize and flatten
        pred = np.expm1(pred*np.log1p(y_max)).flatten()
        # save prediction
        np.save(os.path.join(test_pred_save_dir, file_name), pred)

        # load corresponding label data (ground truth data)
        gt = np.load(os.path.join(y_train_path, file_name)).astype('float32')
        gt = np.reshape(gt, (yn,ym))
        pred = np.reshape(pred, (yn,ym))

        # eval metrics
        mae, rmse, corr, ssim_value = res_eval(pred, gt, mask)
        df['Date'].append(date)
        df['MAE'].append(mae)
        df['RMSE'].append(rmse)
        df['Corr'].append(corr)
        df['SSIM'].append(ssim_value)

        print(f"Exporting pred on {date}...")
    
    # save file
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(save_dir, 'test_eval.csv'))
    with open(os.path.join(save_dir, 'test_eval.txt'), 'a') as f:
        f.write("Test, MAE, RMSE, Corr, SSIM\n")
        f.write(f"AVG, {str(df['MAE'].mean())}, {str(df['RMSE'].mean())}, {str(df['Corr'].mean())}, {str(df['SSIM'].mean())}\n")
        f.write(f"MED, {str(df['MAE'].median())}, {str(df['RMSE'].median())}, {str(df['Corr'].median())}, {str(df['SSIM'].median())}")

if __name__ == '__main__':
    mask = np.reshape(mask, (yn,ym))
    lat = np.linspace(start=25.25, stop=22. , num=yn, endpoint=True)
    lon = np.linspace(start=120. , stop=122., num=ym, endpoint=True)
    val_pred_txt_save_dir  = os.path.join(save_dir,  'val_pred_txt')
    test_pred_txt_save_dir = os.path.join(save_dir, 'test_pred_txt')
    val_pred_paths  = os.path.join(save_dir,  'val_pred')
    test_pred_paths = os.path.join(save_dir, 'test_pred')

    train()
    pred(mask)
    npytotxt(yn=yn, ym=ym, lat=lat, lon=lon, mask=mask, root_dir= val_pred_paths, saveto= val_pred_txt_save_dir)
    npytotxt(yn=yn, ym=ym, lat=lat, lon=lon, mask=mask, root_dir=test_pred_paths, saveto=test_pred_txt_save_dir)
    QoF(mask=mask, pred_dir=save_dir, gt_dir=y_train_path)
