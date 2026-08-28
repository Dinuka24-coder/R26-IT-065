import tensorflow as tf

from app.ml_models.component3.gatekeeper_cnn.config import IMG_SIZE

# Verified directly against tf.keras.applications.MobileNetV3Small(weights='imagenet',
# include_top=False, pooling=None, include_preprocessing=True) at 224x224x3: 157 layers,
# 939,120 params, last spatial feature map is 'conv_1' (7x7x576) before the final
# activation/pooling. Freezing everything through 'expanded_conv_7_*' and unfreezing
# 'expanded_conv_8_*' onward covers the last 3 of 11 inverted-residual blocks (~27%,
# the "last ~30%" fine-tune region train_gatekeeper.py's Phase 2 uses).
UNFREEZE_FROM_LAYER = "expanded_conv_8_expand"

# Grad-CAM target: the 'gap' (GlobalAveragePooling2D) layer's *input* tensor,
# i.e. the backbone's final 7x7x576 feature map (conv_1 -> conv_1_bn ->
# activation_17). Reaching into the backbone submodel's own standalone graph
# by layer name (e.g. backbone.get_layer('conv_1').output) fails in Keras 3
# with "not connected to inputs" because `backbone(inputs)` inside this
# functional model creates a fresh call context distinct from backbone's own
# original graph -- confirmed directly against this TF/Keras build. Using
# 'gap's input sidesteps that: it's the exact tensor produced by the actual
# call in this model's graph, one BN+activation downstream of conv_1, same
# spatial shape, functionally equivalent for Grad-CAM purposes.
GRADCAM_TARGET_LAYER = "gap"


def build_gatekeeper_model():
    """Two-head MobileNetV3Small gatekeeper: cxr_head (is this a chest X-ray)
    and quality_head (is it usable quality), sharing one backbone. See
    training/gatekeeper_cnn/README.md for why two independent sigmoid heads
    were chosen over a single 3-class softmax.

    Input: raw float32 pixels in [0, 255], shape (224, 224, 3) -- the
    backbone's own Rescaling layer (include_preprocessing=True) handles
    normalization internally, so callers must NOT pre-divide by 255. This is
    a deliberate divergence from component3/preprocessing.py's [0,1]
    convention; see gatekeeper_cnn/preprocessing.py.
    """
    backbone = tf.keras.applications.MobileNetV3Small(
        input_shape=IMG_SIZE + (3,),
        weights="imagenet",
        include_top=False,
        pooling=None,
        include_preprocessing=True,
    )
    backbone.trainable = False  # Phase 1 default; train_gatekeeper.py flips this for Phase 2

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,), name="image")
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)

    cxr_branch = tf.keras.layers.Dense(64, activation="relu", name="cxr_dense")(x)
    cxr_branch = tf.keras.layers.Dropout(0.3, name="cxr_dropout")(cxr_branch)
    cxr_head = tf.keras.layers.Dense(1, activation="sigmoid", name="cxr_head")(cxr_branch)

    quality_branch = tf.keras.layers.Dense(64, activation="relu", name="quality_dense")(x)
    quality_branch = tf.keras.layers.Dropout(0.3, name="quality_dropout")(quality_branch)
    quality_head = tf.keras.layers.Dense(1, activation="sigmoid", name="quality_head")(quality_branch)

    model = tf.keras.Model(inputs=inputs, outputs={"cxr_head": cxr_head, "quality_head": quality_head},
                            name="cxr_gatekeeper")
    return model, backbone


def set_finetune_phase(backbone, phase: int):
    """phase=1: freeze the entire backbone (train heads only).
    phase=2: unfreeze from UNFREEZE_FROM_LAYER onward, but keep every
    BatchNormalization layer frozen regardless of phase -- standard practice
    to avoid destabilizing ImageNet-derived running statistics on a dataset
    several orders of magnitude smaller than ImageNet."""
    if phase == 1:
        backbone.trainable = False
        return

    if phase != 2:
        raise ValueError(f"Unknown finetune phase: {phase}")

    backbone.trainable = True
    unfreezing = False
    for layer in backbone.layers:
        if layer.name == UNFREEZE_FROM_LAYER:
            unfreezing = True
        layer.trainable = unfreezing and not isinstance(layer, tf.keras.layers.BatchNormalization)


def compile_gatekeeper_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "cxr_head": tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
            "quality_head": tf.keras.losses.BinaryCrossentropy(),
        },
        loss_weights={"cxr_head": 1.0, "quality_head": 1.0},
        metrics={
            "cxr_head": ["accuracy", tf.keras.metrics.AUC(name="auc"),
                         tf.keras.metrics.Precision(name="precision"),
                         tf.keras.metrics.Recall(name="recall")],
            "quality_head": ["accuracy", tf.keras.metrics.AUC(name="auc"),
                              tf.keras.metrics.Precision(name="precision"),
                              tf.keras.metrics.Recall(name="recall")],
        },
    )
    return model
