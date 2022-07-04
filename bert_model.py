from bert4keras.backend import keras, set_gelu
from bert4keras.models import build_transformer_model
from keras.layers import Lambda, Dense, concatenate
from tensorflow import expand_dims

set_gelu('tanh')


class Model(object):
    # def __init__(self,config_path,checkpoint_path,num_classes):
    #     self.config_path = config_path
    #     self.checkpoint_path = checkpoint_path
    #     self.num_classes = num_classes

    def textcnn(self, inputs, kernel_initializer):
        # 3,4,5
        cnn1 = keras.layers.Conv1D(
            128,
            3,
            strides=1,
            padding='same',
            activation='relu',
            kernel_initializer=kernel_initializer
        )(inputs)  # shape=[batch_size,maxlen-2,128]
        cnn1 = keras.layers.GlobalMaxPooling1D()(cnn1)  # shape=[batch_size,128]

        cnn2 = keras.layers.Conv1D(
            128,
            4,
            strides=1,
            padding='same',
            activation='relu',
            kernel_initializer=kernel_initializer
        )(inputs)
        cnn2 = keras.layers.GlobalMaxPooling1D()(cnn2)

        cnn3 = keras.layers.Conv1D(
            128,
            5,
            strides=1,
            padding='same',
            kernel_initializer=kernel_initializer
        )(inputs)
        cnn3 = keras.layers.GlobalMaxPooling1D()(cnn3)

        output = keras.layers.concatenate(
            [cnn1, cnn2, cnn3],
            axis=-1)
        output = keras.layers.Dropout(0.2)(output)
        return output

    def bert_text_cnn(self, config_path, checkpoint_path, num_classes):
        # 加载预训练模型
        bert = build_transformer_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            model='bert',
            return_keras_model=False,
        )

        cls_features = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)

        all_token_embedding = Lambda(lambda x: x[:, 1:-1], name='all_token')(
            bert.model.output)  # [batch_size,maxlen-2,768]

        cnn_features = self.textcnn(all_token_embedding, bert.initializer)

        concat_features = concatenate([cls_features, cnn_features], axis=-1)

        dense = Dense(
            units=64,
            activation='relu',
            kernel_initializer=bert.initializer
        )(concat_features)

        output = Dense(units=num_classes,
                       activation='softmax',
                       kernel_initializer=bert.initializer)(dense)

        model = keras.models.Model(bert.model.input, output)

        return model

    def bert_dense(self, config_path, checkpoint_path, num_classes):
        # 加载预训练模型
        bert = build_transformer_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            model='bert',
            return_keras_model=False,
        )

        cls_features = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)

        output = Dense(units=num_classes,
                       activation='softmax',
                       kernel_initializer=bert.initializer)(cls_features)

        model = keras.models.Model(bert.model.input, output)

        return model

    def _1D_CNN(self, inputs, kernel_initializer):
        cnn1 = keras.layers.Conv1D(32,
                                   3,
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer=kernel_initializer)(inputs)  # shape=[batch_size,maxlen-2,256]
        cnn1 = keras.layers.GlobalMaxPooling1D()(cnn1)  # shape=[batch_size,256]
        cnn1_add1 = Lambda(lambda x: expand_dims(x, -1))(cnn1)
        cnn2 = keras.layers.Conv1D(64,
                                   3,
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer=kernel_initializer)(cnn1_add1)  # shape=[batch_size,maxlen-2,256]
        cnn2 = keras.layers.GlobalMaxPooling1D()(cnn2)  # shape=[batch_size,256]
        output = keras.layers.Dropout(0.2)(cnn2)

        return output

    def bert_1d_cnn(self, config_path, checkpoint_path, num_classes):
        # 加载预训练模型
        bert = build_transformer_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            model='bert',
            return_keras_model=False,
        )

        cls_features = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)

        all_token_embedding = Lambda(lambda x: x[:, 1:-1], name='all_token')(
            bert.model.output)  # [batch_size,maxlen-2,768]

        cnn_features = self._1D_CNN(all_token_embedding, bert.initializer)

        dense = Dense(
            units=64,
            activation='relu',
            kernel_initializer=bert.initializer
        )(cnn_features)

        output = Dense(units=num_classes,
                       activation='softmax',
                       kernel_initializer=bert.initializer)(dense)

        model = keras.models.Model(bert.model.input, output)

        return model

    def _2D_CNN(self, inputs, kernel_initializer):
        cnn1 = keras.layers.Conv2D(64,
                                   3,
                                   strides=1,
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer=kernel_initializer)(inputs)  # shape=[batch_size,maxlen-2,256]
        cnn1 = keras.layers.MaxPooling2D(2)(cnn1)  # shape=[batch_size,256]
        # cnn1_add1 = Lambda(lambda x: expand_dims(x, -1))(cnn1)
        cnn2 = keras.layers.Conv2D(32,
                                   3,
                                   strides=1,
                                   padding='same',
                                   activation='relu',
                                   kernel_initializer=kernel_initializer)(cnn1)
        cnn2 = keras.layers.GlobalMaxPooling2D()(cnn2)  # 可以代替flatten层
        output = keras.layers.Dropout(0.2)(cnn2)

        return output

    def bert_2d_cnn(self, config_path, checkpoint_path, num_classes):
        # 加载预训练模型
        bert = build_transformer_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            model='bert',
            return_keras_model=False,
        )

        cls_features = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)

        all_token_embedding = Lambda(lambda x: x[:, 1:-1], name='all_token')(
            bert.model.output)  # [batch_size,maxlen-2,768]

        all = Lambda(lambda x: expand_dims(x, -1))(all_token_embedding)

        cnn_features = self._2D_CNN(all, bert.initializer)

        dense = Dense(
            units=64,
            activation='relu',
            kernel_initializer=bert.initializer
        )(cnn_features)

        output = Dense(units=num_classes,
                       activation='softmax',
                       kernel_initializer=bert.initializer)(dense)

        model = keras.models.Model(bert.model.input, output)

        return model
