import pandas
import numpy
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from lookout.style.typos_checker.typos_correction.utils import collect_embeddings


def get_features(fasttext, typos) -> numpy.ndarray:
    return numpy.concatenate((collect_embeddings(fasttext, typos),
                             numpy.ones((len(typos), 1))), axis=1)


def get_target(fasttext, identifiers) -> numpy.ndarray:
    return collect_embeddings(fasttext, identifiers)


def create_model(input_len: int, output_len: int) -> keras.models.Sequential:
    model = Sequential()
    model.add(Dense(units=256, activation='relu', input_dim=input_len))
    model.add(Dense(units=256, activation='relu'))
    model.add(Dense(units=output_len))
    return model


def generator(features: numpy.ndarray, target, batch_size: numpy.ndarray):
    while True:
        indices = numpy.random.randint(features.shape[0], size=batch_size)
        batch_features = features[indices]
        batch_target = target[indices]
        yield batch_features, batch_target


def train_model(model: keras.models.Sequential, features: numpy.ndarray, target: numpy.ndarray,
                save_model_file: str = "best_nn_prediction.h5", batch_size: int = 64, lr: float = 0.1,
                decay: float = 1e-7, num_epochs: int = 100) -> keras.models.Sequential:
    train_features, val_features, train_target, val_target = train_test_split(features, target,
                                                                              test_size=0.005,
                                                                              random_state=42)

    model.compile(optimizer=keras.optimizers.SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True),
                  loss='cosine_proximity')
    model.fit_generator(generator(train_features, train_target, batch_size=batch_size),
                        steps_per_epoch=len(features) // batch_size, epochs=num_epochs,
                        validation_data=(val_features, val_target),
                        callbacks=[ModelCheckpoint(save_model_file, monitor='val_loss',
                                                   verbose=1, save_best_only=True, mode='max')])
    return model


def create_and_train_model(fasttext, data: pandas.DataFrame,
                           save_model_file: str = None,
                           batch_size: int = 64, lr: float = 0.1, decay: float = 1e-7,
                           num_epochs: int = 100) -> keras.models.Sequential:
    typo_vecs = get_features(fasttext, data.typo)
    correction_vecs = get_target(fasttext, data.identifier)
    
    model = create_model(typo_vecs.shape[1], correction_vecs.shape[1])

    train_model(model, typo_vecs, correction_vecs, save_model_file, batch_size, lr,
                decay, num_epochs)
    return model


def get_predictions(fasttext, model: keras.models.Sequential, typos: pandas.Series):
    return model.predict(get_features(fasttext, typos))
