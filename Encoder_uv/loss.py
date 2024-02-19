import tensorflow as tf
# loss function
class wmse(tf.keras.losses.Loss):
    def __init__(self, wmse_gamma):
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