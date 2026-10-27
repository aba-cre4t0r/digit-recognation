"""Draw a digit with the mouse and let the trained model predict it.
Controls: hold left mouse = draw | ENTER = predict | C = clear
Run `python train.py` first to create digit_model.keras.
"""
import numpy as np
import pygame
import tensorflow as tf
from scipy.ndimage import center_of_mass

model = tf.keras.models.load_model("digit_model.keras")

pygame.init()
screen = pygame.display.set_mode((280, 280))
pygame.display.set_caption("Draw a Digit (ENTER=predict, C=clear)")
font = pygame.font.SysFont("Arial", 24)


def preprocess(surface):
    """Turn the 280x280 drawing into an MNIST-style 28x28 input."""
    data = pygame.surfarray.array3d(surface)
    gray = np.mean(data, axis=2).T

    # crop to the bounding box of the drawn digit
    rows, cols = np.any(gray > 10, axis=1), np.any(gray > 10, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    cropped = np.pad(gray[rmin:rmax + 1, cmin:cmax + 1], 30, constant_values=0)

    # resize to 20x20, place in a 28x28 canvas (MNIST style)
    rgb = np.stack([cropped] * 3, axis=-1).astype(np.uint8)
    surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
    small = np.mean(pygame.surfarray.array3d(
        pygame.transform.smoothscale(surf, (20, 20))), axis=2).T
    canvas = np.zeros((28, 28))
    canvas[4:24, 4:24] = small / 255.0

    # shift so the center of mass sits in the middle
    cy, cx = center_of_mass(canvas)
    canvas = np.roll(canvas, int(14 - cy), axis=0)
    return np.roll(canvas, int(14 - cx), axis=1)


text = ""
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                img = preprocess(screen)
                if img is not None:
                    pred = model.predict(img.reshape(1, 28, 28, 1).astype(np.float32), verbose=0)
                    text = f"Digit: {np.argmax(pred)}  ({np.max(pred) * 100:.1f}%)"
                else:
                    text = "Nothing drawn!"
            if event.key == pygame.K_c:
                screen.fill((0, 0, 0))
                text = ""

    if pygame.mouse.get_pressed()[0]:
        pygame.draw.circle(screen, (255, 255, 255), pygame.mouse.get_pos(), 12)

    pygame.draw.rect(screen, (0, 0, 0), (0, 0, 280, 35))  # clear text strip
    screen.blit(font.render(text, True, (0, 255, 0)), (5, 5))
    pygame.display.flip()

pygame.quit()
