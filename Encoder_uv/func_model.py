from tensorflow.keras.layers import Input, Layer, MultiHeadAttention
from tensorflow.keras.layers import LayerNormalization, Dropout
from tensorflow.keras.layers import  Dense, Conv2D, Lambda
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import Model
from tensorflow.nn import depth_to_space
from tensorflow import reshape, shape, repeat, range
'''
    Write model layers in a function way.    
'''
def SelfAttentionLayer(input, d_model, num_heads, dff, dropout_rate=0.2):
    # Multi-head self-attention
    x = MultiHeadAttention(num_heads=num_heads,
                           key_dim=d_model)(input, input, input)
    x = Dropout(dropout_rate)(x)
    out1 = LayerNormalization(epsilon=1e-6)(x + input)

    x = Dense(dff, activation='relu')(out1)
    x = Dense(d_model)(x)
    x = Dropout(dropout_rate)(x)
    x = LayerNormalization(epsilon=1e-6)(x + out1)
    return x

class TrainablePositionalEncoding(Layer):
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
        input_shape = shape(inputs)
        batch_size, seq_length = input_shape[0], input_shape[1]

        # Slice positional encoding to match the input sequence length
        pos_encoding = self.positional_encoding[:, :seq_length, :]

        return inputs + pos_encoding

def Encoder(num_layers, vector_length, channel, input_embed_size,
            d_model, dff, num_heads, xn, xm, dropout_rate=0.2):
    '''
        N: number of layers of self-attention modules.
        vector_length: flattened grid points.
        channel: number of climate variables.
        input_embed_size: size of input embedding, like word embedding size in Word2Vec.
        max_length: positional embedding size.
        d_model: dim of the embeddings throughout the model.
        dff: dim of dense feedforward layer.
        num_heads: number of heads of attention layer.
        xn, xm: 2D targe height and width.
    '''
    # vector length: flattened 2D data => 14x9 to 126
    # channel: number of multi-variable
    x = Input(shape = (vector_length, channel))
    x0 = x

    # input embedding
    x = Dense(input_embed_size)(x)
    print("X Dense Shape: ", x.shape)
    # positional encoding
    x = TrainablePositionalEncoding(input_embed_size, d_model)(x)

    # Self-attention Layers
    for i in range(num_layers):
        x = SelfAttentionLayer(input=x, d_model=d_model,
                               num_heads=num_heads, dff=dff,
                               dropout_rate=dropout_rate)
    
    # Encoder output layer
    x = Dense(1)(x)
    x = reshape(x, [-1, xn, xm, 1]) # 2D

    return Model(inputs=x0, outputs=x)

def Resolver(xn, xm, channel, scale, topo):
    '''
        xn, xm: 2D targe height and width.
        channel: number of climate variables.
        scale: downscaling factor (image upscaling factor)
        topo: topology data
    '''
    x = Input(shape = (xn, xm, channel))
    x0 = x

    x = Conv2D(filters=scale**2, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = Lambda(lambda x: depth_to_space(x, scale))(x) # subpixel
    x = Conv2D(filters=1, kernel_size=(3,3), padding='same', activation='relu')(x)
    if topo is not None:
        batch = shape(x)[0]
        topo = repeat(topo, batch, axis=0)
        x = Concatenate()([x, topo]) # concat with topology
    x = Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = Conv2D(filters=1, kernel_size=(3,3), padding='same', activation='relu')(x)

    return Model(inputs=x0, outputs=x)