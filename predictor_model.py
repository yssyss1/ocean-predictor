from keras.layers import Input, ConvLSTM2D, Conv2D, BatchNormalization, Conv3D, Lambda
from keras.models import Model
import keras.backend as K


def ae_convlstm_model(input_shape, output_channel):
    """
    AutoEncoder + Convolutional LSTM 구조
    :param input_shape: 입력 크기
    :param output_channel: 출력 크기
    """

    input = Input(shape=input_shape[1:])
    input_reshape = Lambda(lambda x: K.permute_dimensions(x, (0, 4, 2, 3, 1)))(input)

    encoder_1 = Conv3D(filters=input_shape[1] // 2, kernel_size=(1, 5, 5), padding='same', activation='relu')(
        input_reshape)
    encoder_1 = BatchNormalization()(encoder_1)
    encoder_2 = Conv3D(filters=input_shape[1] // 4, kernel_size=(1, 5, 5), padding='same', activation='relu')(encoder_1)

    decoder_1 = Conv3D(filters=input_shape[1] // 2, kernel_size=(1, 5, 5), padding='same', activation='relu')(encoder_2)
    decoder_1 = BatchNormalization()(decoder_1)
    decoder_2 = Conv3D(filters=input_shape[1], kernel_size=(1, 5, 5), padding='same', activation='sigmoid')(decoder_1)

    decoder_2_reshape = Lambda(lambda x: K.permute_dimensions(x, (0, 4, 2, 3, 1)))(decoder_2)

    conv_lstm = ConvLSTM2D(filters=12,
                           kernel_size=(5, 5),
                           padding='same',
                           return_sequences=True,
                           stateful=False)(decoder_2_reshape)
    conv_lstm = BatchNormalization()(conv_lstm)

    conv_lstm = ConvLSTM2D(filters=12,
                           kernel_size=(5, 5),
                           padding='same',
                           return_sequences=False,
                           stateful=False)(conv_lstm)

    output = Conv2D(filters=output_channel,
                    kernel_size=(5, 5),
                    activation='sigmoid',
                    padding='same')(conv_lstm)

    return Model(input, output)


def naive_convlstm_model(input_shape, output_channel):
    """
    Convolutional LSTM 2개 쌓은 모델
    :param input_shape: 입력 크기
    :param output_channel: 입력 크기
    """

    input = Input(shape=input_shape[1:])

    conv_lstm = ConvLSTM2D(filters=32,
                           kernel_size=(8, 8),
                           padding='same',
                           return_sequences=False,
                           stateful=False)(input)
    conv_lstm = BatchNormalization()(conv_lstm)

    output = Conv2D(filters=6,
                    kernel_size=(8, 8),
                    activation='relu',
                    padding='same')(conv_lstm)

    output = BatchNormalization()(output)

    output = Conv2D(filters=output_channel,
                    kernel_size=(8, 8),
                    activation='sigmoid',
                    padding='same')(output)

    return Model(input, output)
