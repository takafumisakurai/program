NNモデル
def build_network(self):
    c = input_ = Input(shape=(self.input_sequence,) + self.input_shape)

    if self.enable_image_layer:
        # (time steps, w, h) -> (time steps, w, h, ch)
        c = Reshape((self.input_sequence, ) + self.input_shape + (1,) )(c)

        c = TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4), padding="same"),name="c1")(c)
        c = Activation("relu")(c)
        c = TimeDistributed(Conv2D(64, (4, 4), strides=(2, 2), padding="same"),name="c2")(c)
        c = Activation("relu")(c)
        c = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), padding="same"),name="c3")(c)
        c = Activation("relu")(c)

        c = TimeDistributed(Flatten())(c)

    c = LSTM(self.lstm_units_num)(c)

    if self.enable_dueling_network:
        # value
        v = Dense(self.dense_units_num, activation="relu")(c)
        v = Dense(1, activation="relu", name="v")(v)

        # advance
        adv = Dense(self.dense_units_num, activation='relu')(c)
        adv = Dense(self.nb_actions, name="adv")(adv)

        # 連結で結合
        c = Concatenate()([v,adv])
        c = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], axis=1, keepdims=True), output_shape=(self.nb_actions,))(c)

    else:
        c = Dense(self.dense_units_num, activation="relu")(c)
        c = Dense(self.nb_actions, activation="linear")(c)

    return Model(input_, c)
