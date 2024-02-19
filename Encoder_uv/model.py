import tensorflow as tf


# positional encoding based on height and width
# class ImagePositionalEncoding(tf.keras.layers.Layer):
#     def __init__(self, image_height, image_width, channel):
#         super(ImagePositionalEncoding, self).__init__()
#         self.image_height = image_height
#         self.image_width = image_width
#         self.channel = channel

#     def get_positional_encoding(self):
#         # Create positional encoding for each position in the image
#         pos_encoding = tf.convert_to_tensor([
#             [pos / 10000 ** (2 * (j // 2) / self.channel) for j in range(self.channel)]
#             if pos % 2 == 0 else
#             [pos / 10000 ** ((2 * (j - 1)) / self.channel) for j in range(self.channel)]
#             for pos in range(self.image_height * self.image_width)
#         ])
#         pos_encoding = tf.convert_to_tensor(pos_encoding)
#         pos_encoding = tf.reshape(pos_encoding, (1, self.image_height, self.image_width, self.channel))
#         return pos_encoding

#     def call(self, inputs):
#         batch_size = tf.shape(inputs)[0]
#         pos_encoding = self.get_positional_encoding()

#         # Tile positional encoding to match the batch size
#         pos_encoding = tf.tile(pos_encoding, [batch_size, 1, 1, 1])

#         return inputs + pos_encoding

class TrainablePositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_length, d_model):
        super(TrainablePositionalEncoding, self).__init__()
        self.max_length = max_length
        self.d_model = d_model

        self.positional_encoding = self.add_weight(
            "positional_encoding",
            shape=(1, self.max_length, self.d_model),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, seq_length = input_shape[0], input_shape[1]

        # Slice positional encoding to match the input sequence length
        pos_encoding = self.positional_encoding[:, :seq_length, :]

        return inputs + pos_encoding

# Define the Encoder layer with self-attention
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.2):
        super(EncoderLayer, self).__init__()

        # Multi-head self-attention
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)

        # Feedforward layer
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])

        # Layer normalization
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # Dropout layers
        self.dropout1 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, x, training, mask=None):
        # Multi-head self-attention
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Feedforward layer
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

# Define the Encoder
class Encoder(tf.keras.Model):
    def __init__(self, xn, xm, num_layers, d_model, num_heads, dff, input_vocab_size, channel,
                 dropout_rate=0.2, reshape=True):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.input_vocab_size = input_vocab_size
        self.channel = channel
        self.xn = xn
        self.xm = xm
        self.reshape = reshape
        self.embedding = tf.keras.layers.Dense(dff)
        self.pos_encoding = TrainablePositionalEncoding(input_vocab_size, d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate) for _ in range(num_layers)]
        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, x, training, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)
        x = self.output_layer(x)
        if self.reshape:
            x = tf.reshape(x, [-1, self.xn, self.xm, 1]) # 2D
        return x

class Resovler(tf.keras.Model):
    def __init__(self, scale, topo=None):
        super(Resovler, self).__init__()
        self.scale = scale
        self.topo = topo # 4D topology array (1,H,W,1)
        self.conv1 = tf.keras.layers.Conv2D(filters=scale**2, kernel_size=(3,3), padding='same', activation='relu')
        self.upsample = tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, scale)) # subpixel
        self.conv2 = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), padding='same', activation='relu')
        self.concat = tf.keras.layers.Concatenate() # concat with topology
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')
        self.output_layer = tf.keras.layers.Conv2D(filters=1, kernel_size=(3,3), padding='same', activation='relu')
    
    def call(self, x):
        x = self.conv1(x)
        x = self.upsample(x)
        x = self.conv2(x)
        if self.topo is not None:
            batch = tf.shape(x)[0]
            topo = tf.repeat(self.topo, batch, axis=0)
            x = self.concat([x, topo])
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.output_layer(x)

        return x