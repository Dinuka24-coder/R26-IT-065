import tensorflow as tf
import os

UNET_PATH = os.path.join(
    os.path.dirname(__file__), "weights", "pneumothorax_unet.keras"
)

unet_model = None

# Custom objects needed to load the model
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )
    return 1 - dice

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(tf.cast(y_pred > 0.5, tf.float32), [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

def get_unet_model():
    global unet_model
    if unet_model is None:
        unet_model = tf.keras.models.load_model(
            UNET_PATH,
            custom_objects={
                "combined_loss":     combined_loss,
                "dice_loss":         dice_loss,
                "dice_coefficient":  dice_coefficient,
            }
        )
        print("✅ U-Net segmentation model loaded")
    return unet_model