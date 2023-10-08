
def multiscale_squeeze_excite_block(inputs, ratio=16, scales=[1, 2, 3, 4, 5, 6]):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]

    se_branches = []
    for scale in scales:
        # Reduce input dimensions for each scale
        reduced_input = Conv2D(filters // scale, kernel_size=1, activation='relu')(init)

        # Apply Squeeze-and-Excite mechanism
        se = GlobalAveragePooling2D()(reduced_input)
        se = Reshape((1, 1, filters // scale))(se)
        se = Dense((filters // scale) // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters // scale, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

        # Expand back to original dimensions
        se = Conv2D(filters, kernel_size=1, activation='sigmoid')(se)

        se_branches.append(se)

    multiscale_se = se_branches[0]
    for se in se_branches[1:]:
        multiscale_se = Multiply()([multiscale_se, se])

    x = Multiply()([init, multiscale_se])
    return x


