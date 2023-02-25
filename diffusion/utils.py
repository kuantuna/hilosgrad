def normalize(img):
    return img / 255 * 2 - 1


def denormalize(img):
    return (img + 1) / 2
