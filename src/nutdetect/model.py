import keras
from keras import layers
from keras.applications import MobileNetV2

BACKBONE = "backbone"


def augmentation():
    # no flips: a mirrored nut looks like the "flip" defect
    return keras.Sequential([
        layers.RandomRotation(0.5, fill_mode="reflect"),
        layers.RandomTranslation(0.08, 0.08, fill_mode="reflect"),
        layers.RandomZoom(0.1, fill_mode="reflect"),
        layers.RandomBrightness(0.1, value_range=(0, 255)),
        layers.RandomContrast(0.1),
    ], name="augment")


def build_model(img_size, weights="imagenet"):
    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = augmentation()(inputs)
    x = layers.Rescaling(1 / 127.5, offset=-1)(x)

    backbone = MobileNetV2(
        include_top=False, weights=weights,
        input_shape=(img_size, img_size, 3), name=BACKBONE,
    )
    backbone.trainable = False
    x = backbone(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid", name="p_defect")(x)
    return keras.Model(inputs, out)


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.Precision(name="precision"),
        ],
    )


def unfreeze_top(model, n):
    backbone = model.get_layer(BACKBONE)
    backbone.trainable = True
    for layer in backbone.layers[:-n]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
