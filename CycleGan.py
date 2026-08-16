# Clean, Fast CycleGAN (Horse ↔ Zebra) - Jupyter/VS Code Compatible
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
import matplotlib.pyplot as plt
import os
import time
from IPython.display import display

AUTOTUNE = tf.data.AUTOTUNE

print("Loading cycle_gan/horse2zebra dataset...")
dataset, metadata = tfds.load(
    'cycle_gan/horse2zebra',
    with_info=True,
    as_supervised=True
)

train_horses = dataset['trainA']
train_zebras = dataset['trainB']
test_horses = dataset['testA']
test_zebras = dataset['testB']

# ------------ Hyperparameters -------------
BUFFER_SIZE = 200
BATCH_SIZE = 1
IMG_HEIGHT, IMG_WIDTH = 256, 256
LAMBDA = 5

EPOCHS = 10            #  Increased for higher accuracy
MAX_STEPS = 1000       #  Increased steps for stronger training

os.makedirs("samples", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ------------ Preprocessing ----------------
def normalize(img):
    return (tf.cast(img, tf.float32) / 127.5) - 1

def preprocess_train(img, label):
    img = tf.image.resize(img, [286, 286])
    img = tf.image.random_crop(img, [IMG_HEIGHT, IMG_WIDTH, 3])
    img = tf.image.random_flip_left_right(img)
    return normalize(img)

def preprocess_test(img, label):
    return normalize(img)

train_horses = train_horses.map(preprocess_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_zebras = train_zebras.map(preprocess_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_horses = test_horses.map(preprocess_test).batch(BATCH_SIZE)
test_zebras = test_zebras.map(preprocess_test).batch(BATCH_SIZE)

sample_horse = next(iter(train_horses))
sample_zebra = next(iter(train_zebras))

# ------------ Models -----------------------
generator_g = pix2pix.unet_generator(3, norm_type='instancenorm')  # horse → zebra
generator_f = pix2pix.unet_generator(3, norm_type='instancenorm')  # zebra → horse
discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)

_ = generator_g(sample_horse)
_ = generator_f(sample_zebra)
_ = discriminator_x(sample_horse)
_ = discriminator_y(sample_zebra)

# ------------ Losses -----------------------
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def gen_loss(fake):
    return loss_obj(tf.ones_like(fake), fake)

def disc_loss(real, fake):
    return (loss_obj(tf.ones_like(real), real) +
            loss_obj(tf.zeros_like(fake), fake)) * 0.5

def cycle_loss(real, cycled):
    return LAMBDA * tf.reduce_mean(tf.abs(real - cycled))

def identity_loss(real, same):
    return 0.5 * LAMBDA * tf.reduce_mean(tf.abs(real - same))

# ------------ Optimizers -------------------
opt_g = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
opt_f = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
opt_dx = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
opt_dy = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# ------------ Training Step ----------------
@tf.function
def train_step(real_x, real_y):
    with tf.GradientTape(persistent=True) as tape:
        fake_y = generator_g(real_x, training=True)
        fake_x = generator_f(real_y, training=True)

        cycled_x = generator_f(fake_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        g_loss = gen_loss(disc_fake_y)
        f_loss = gen_loss(disc_fake_x)

        cyc_loss = cycle_loss(real_x, cycled_x) + cycle_loss(real_y, cycled_y)

        total_g = g_loss + cyc_loss + identity_loss(real_y, same_y)
        total_f = f_loss + cyc_loss + identity_loss(real_x, same_x)

        dx_loss = disc_loss(disc_real_x, disc_fake_x)
        dy_loss = disc_loss(disc_real_y, disc_fake_y)

    opt_g.apply_gradients(zip(tape.gradient(total_g, generator_g.trainable_variables), generator_g.trainable_variables))
    opt_f.apply_gradients(zip(tape.gradient(total_f, generator_f.trainable_variables), generator_f.trainable_variables))
    opt_dx.apply_gradients(zip(tape.gradient(dx_loss, discriminator_x.trainable_variables), discriminator_x.trainable_variables))
    opt_dy.apply_gradients(zip(tape.gradient(dy_loss, discriminator_y.trainable_variables), discriminator_y.trainable_variables))

# ------------ Utilities --------------------
def deprocess(img):
    return tf.cast((img + 1) * 127.5, tf.uint8).numpy()

def show_result(model, inp, title="Result"):
    pred = model(inp, training=False)
    inp_img = deprocess(inp[0])
    pred_img = deprocess(pred[0])
    combined = tf.concat([inp_img, pred_img], axis=1)
    plt.figure(figsize=(8, 4))
    plt.imshow(combined)
    plt.axis("off")
    plt.title(title)
    display(plt.gcf())
    plt.close()

def save_result(model, inp, fname):
    pred = model(inp, training=False)
    merged = tf.concat([deprocess(inp[0]), deprocess(pred[0])], axis=1)
    plt.imsave(fname, merged)

# ------------ Training Loop ----------------
print("Training...")

for epoch in range(EPOCHS):
    start = time.time()
    step = 0

    for real_x, real_y in tf.data.Dataset.zip((train_horses, train_zebras)):
        train_step(real_x, real_y)
        step += 1

        if step % 50 == 0:
            print(f"Epoch {epoch+1} | Step {step}")

        if step >= MAX_STEPS:
            break

    save_result(generator_g, sample_horse, f"samples/e{epoch+1}_AtoB.png")
    save_result(generator_f, sample_zebra, f"samples/e{epoch+1}_BtoA.png")

    show_result(generator_g, sample_horse, title=f"Epoch {epoch+1}: Horse → Zebra")
    show_result(generator_f, sample_zebra, title=f"Epoch {epoch+1}: Zebra → Horse")

    print(f"Epoch {epoch+1} finished in {time.time() - start:.1f} sec")

# ------------ Display test set results -----------
print("Displaying some test set results...")

test_horse_sample = next(iter(test_horses))
test_zebra_sample = next(iter(test_zebras))

show_result(generator_g, test_horse_sample, title="Test: Horse → Zebra")
show_result(generator_f, test_zebra_sample, title="Test: Zebra → Horse")

# ------------ SAVE MODELS ------------------
print("Saving trained models...")

generator_g.save("models/generator_horse_to_zebra.h5")
generator_f.save("models/generator_zebra_to_horse.h5")
discriminator_x.save("models/discriminator_x.h5")
discriminator_y.save("models/discriminator_y.h5")

print("Models saved in /models")
print("Done!")
