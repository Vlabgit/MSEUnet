def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def patch_dice(predicted_image, true_image):
    patch_size=(16, 16)
    stride=(16, 16)
    # Extract the patches from both predicted and true images
    predicted_patches = tf.image.extract_patches(
        predicted_image,
        sizes=[1, patch_size[0], patch_size[1], 1],
        strides=[1, stride[0], stride[1], 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    true_patches = tf.image.extract_patches(
        true_image,
        sizes=[1, patch_size[0], patch_size[1], 1],
        strides=[1, stride[0], stride[1], 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )

    # Calculate dice loss for each patch
    dice_loss_patch = dice_loss(predicted_patches, true_patches)

    # Calculate the max of patch-wise Dice loss
    max_dice_loss = tf.reduce_max(dice_loss_patch)

    return max_dice_loss


#def dicepatch_bce(y_true,y_pred):
#     bce=tf.keras.losses.binary_crossentropy(y_true, y_pred)
#     dice = patch_dice(y_true, y_pred)
#     return dice+bce
