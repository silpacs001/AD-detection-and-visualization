##### VISUALIZATION ##########

from torchvision.transforms import (CenterCrop,
                                    Compose,
                                    Normalize,
                                    RandomRotation,
                                    RandomResizedCrop,
                                    RandomHorizontalFlip,
                                    RandomAdjustSharpness,
                                    Resize,
                                    ToTensor)

normalize = Normalize(mean=0.5, std=0.5)
_train_transforms = Compose(
        [
            Resize((224, 224)),
            RandomRotation(15),
            RandomAdjustSharpness(2),
            ToTensor(),
            normalize,
        ]
    )

def apply_transform(img: Image) -> np.ndarray:

    transformed_image = _train_transforms(img)
    img_array = transformed_image.numpy().transpose((1, 2, 0))
    img_array = np.clip(img_array, 0, 1)

    return img_array


def visualize_transform(image: np.ndarray, original_image: np.ndarray = None) -> None:

    fontsize = 18

    if original_image is None:
        f, ax = plt.subplots(1, 2, figsize=(12, 12))
        ax[0].imshow(image)
    else:
        f, ax = plt.subplots(1, 2, figsize=(12, 12))
        ax[0].imshow(original_image)
        ax[0].set_title('Original image', fontsize=fontsize)
        ax[1].imshow(image)
        ax[1].set_title('Transformed image', fontsize=fontsize)

img = Image.open('fig_AD.jpg')
img_array = apply_transform(img)
visualize_transform(img_array, original_image=img)

#Image to Flattened Image Patches
import tensorflow as tf

def read_image(image_file="fig_AD.jpg", scale=False, image_dim=159):

    image = tf.keras.utils.load_img(
        image_file, grayscale=False, color_mode='rgb', target_size=None,
        interpolation='nearest'
    )
    image_arr_orig = tf.keras.preprocessing.image.img_to_array(image)
    if(scale):
        image_arr_orig = tf.image.resize(
            image_arr_orig, [image_dim, image_dim],
            method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False
        )
    image_arr = tf.image.crop_to_bounding_box(
        image_arr_orig, 0, 0, image_dim, image_dim
    )

    return image_arr


def create_patches(image):
    im = tf.expand_dims(image, axis=0)
    patches = tf.image.extract_patches(
        images=im,
        sizes=[1, 16, 16, 1],
        strides=[1, 16, 16, 1],
        rates=[1, 1, 1, 1],
        padding="VALID"
    )
    patch_dims = patches.shape[-1]
    patches = tf.reshape(patches, [1, -1, patch_dims])

    return patches

image_arr = read_image()
patches = create_patches(image_arr)


import numpy as np
import matplotlib.pyplot as plt

def render_image_and_patches(image, patches):
    plt.figure(figsize=(8, 8))
    plt.suptitle(f"Cropped Image", size=24)
    plt.imshow(tf.cast(image, tf.uint8))
    plt.axis("off")
    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(8, 8))
    plt.suptitle(f"Image Patches", size=24)
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i+1)
        patch_img = tf.reshape(patch, (16, 16, 3))
        ax.imshow(patch_img.numpy().astype("uint8"))
        ax.axis("off")

def render_flat(patches):
    plt.figure(figsize=(32, 2))
    plt.suptitle(f"Flattened Image Patches", size=24)
    n = int(np.sqrt(patches.shape[1]))
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(1, 24, i+1)
        patch_img = tf.reshape(patch, (16, 16, 3))
        ax.imshow(patch_img.numpy().astype("uint8"))
        ax.axis("off")
        if(i == 23):
            break


render_image_and_patches(image_arr, patches)
render_flat(patches)

#Linear projection and position encoding

NUM_PATCHES = 9 #16
PROJECTION_DIM = 3 #4

image_arr = read_image(image_file="fig_AD.jpg", scale=True, image_dim=64)
patches = create_patches(image_arr)

positions = tf.range(start=0, limit=NUM_PATCHES, delta=1)
projection = tf.keras.layers.Dense(units=PROJECTION_DIM)(patches)
position_embedding = tf.keras.layers.Embedding(input_dim=NUM_PATCHES, output_dim=PROJECTION_DIM)(positions)

final_embedding = projection + position_embedding

orig_size = np.prod(patches.shape)
size_representation = np.prod(final_embedding.shape)
print(f"Shape of patches: {patches.shape} ⇒ {orig_size}, Shape of the final embedding: {final_embedding.shape} ⇒ {size_representation}")
print(f"1:{abs(orig_size / size_representation)} time reduced")
render_image_and_patches(image_arr, patches)
render_flat(patches)

import zipfile
from io import BytesIO
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import requests
import tensorflow as tf
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

RESOLUTION = 224
PATCH_SIZE = 16

crop_layer = keras.layers.CenterCrop(RESOLUTION, RESOLUTION)
norm_layer = keras.layers.Normalization(
    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    variance=[(0.229 * 255) ** 2, (0.224 * 255) ** 2, (0.225 * 255) ** 2],
)
rescale_layer = keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1)


def preprocess_image(image, model_type, size=RESOLUTION):
    image = np.array(image)
    image = tf.expand_dims(image, 0)

    if model_type == "original_vit":
        image = rescale_layer(image)


    resize_size = int((256 / 224) * size)
    image = tf.image.resize(image, (resize_size, resize_size), method="bicubic")

    # Crop the image.
    image = crop_layer(image)

    if model_type != "original_vit":
        image = norm_layer(image)

    return image.numpy()


def load_image_from_url(url, model_type):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    preprocessed_image = preprocess_image(image, model_type)
    return image, preprocessed_image


image = Image.open('fig_AD.jpg')
preprocessed_image = preprocess_image(image,model_type="original_vit")
plt.imshow(image)
plt.axis("off")
plt.show()

def load_model(model_path: str) -> tf.keras.Model:
    with zipfile.ZipFile(model_path, "r") as zip_ref:
        zip_ref.extractall("Probing_ViTs/")
    model_name = model_path.split(".")[0]

    inputs = keras.Input((RESOLUTION, RESOLUTION, 3))
    model = keras.models.load_model(model_name, compile=False)
    outputs, attention_weights = model(inputs, training=False)

    return keras.Model(inputs, outputs=[outputs, attention_weights])

vit_base_i21k_patch16_224 = load_model(MODELS_ZIP["vit_b16_patch16_224-i1k_pretrained"])
print("Model loaded.")

predictions, attention_score_dict = vit_base_i21k_patch16_224.predict(
    preprocessed_image
)
predicted_label = imagenet_int_to_str[int(np.argmax(predictions))]
print(predicted_label)

def compute_distance_matrix(patch_size, num_patches, length):
    distance_matrix = np.zeros((num_patches, num_patches))
    for i in range(num_patches):
        for j in range(num_patches):
            if i == j:  # zero distance
                continue

            xi, yi = (int(i / length)), (i % length)
            xj, yj = (int(j / length)), (j % length)
            distance_matrix[i, j] = patch_size * np.linalg.norm([xi - xj, yi - yj])

    return distance_matrix


def compute_mean_attention_dist(patch_size, attention_weights, model_type):
    num_cls_tokens = 2 if "distilled" in model_type else 1

    attention_weights = attention_weights[
        ..., num_cls_tokens:, num_cls_tokens:
    ]
    num_patches = attention_weights.shape[-1]
    length = int(np.sqrt(num_patches))
    assert length**2 == num_patches, "Num patches is not perfect square"

    distance_matrix = compute_distance_matrix(patch_size, num_patches, length)
    h, w = distance_matrix.shape

    distance_matrix = distance_matrix.reshape((1, 1, h, w))
    mean_distances = attention_weights * distance_matrix
    mean_distances = np.sum(
        mean_distances, axis=-1
    )
    mean_distances = np.mean(
        mean_distances, axis=-1
    )

    return mean_distances

mean_distances = {
    f"{name}_mean_dist": compute_mean_attention_dist(
        patch_size=PATCH_SIZE,
        attention_weights=attention_weight,
        model_type="original_vit",
    )
    for name, attention_weight in attention_score_dict.items()
}

num_heads = tf.shape(mean_distances["transformer_block_0_att_mean_dist"])[-1].numpy()

# Print the shapes
print(f"Num Heads: {num_heads}.")

plt.figure(figsize=(9, 9))

for idx in range(len(mean_distances)):
    mean_distance = mean_distances[f"transformer_block_{idx}_att_mean_dist"]
    x = [idx] * num_heads
    y = mean_distance[0, :]
    plt.scatter(x=x, y=y, label=f"transformer_block_{idx}")

plt.legend(loc="lower right")
plt.xlabel("Attention Head", fontsize=14)
plt.ylabel("Attention Distance", fontsize=14)
plt.title("vit_base_i21k_patch16_224", fontsize=14)
plt.grid()
plt.show()

#Attention heatmaps
# Load the model.
vit_dino_base16 = load_model(MODELS_ZIP["vit_dino_base16"])
print("Model loaded.")
preprocessed_image = preprocess_image(image, model_type="dino")

# Grab the predictions.
predictions, attention_score_dict = vit_dino_base16.predict(preprocessed_image)


def attention_heatmap(attention_score_dict, image, model_type="dino"):
    num_tokens = 2 if "distilled" in model_type else 1

    attention_score_list = list(attention_score_dict.keys())
    attention_score_list.sort(key=lambda x: int(x.split("_")[-2]), reverse=True)

    w_featmap = image.shape[2] // PATCH_SIZE
    h_featmap = image.shape[1] // PATCH_SIZE
    attention_scores = attention_score_dict[attention_score_list[0]]
    attentions = attention_scores[0, :, 0, num_tokens:].reshape(num_heads, -1)

    attentions = attentions.reshape(num_heads, w_featmap, h_featmap)
    attentions = attentions.transpose((1, 2, 0))
    attentions = tf.image.resize(
        attentions, size=(h_featmap * PATCH_SIZE, w_featmap * PATCH_SIZE)
    )
    return attentions


in1k_mean = tf.constant([0.485 * 255, 0.456 * 255, 0.406 * 255])
in1k_std = tf.constant([0.229 * 255, 0.224 * 255, 0.225 * 255])
preprocessed_img_orig = (preprocessed_image * in1k_std) + in1k_mean
preprocessed_img_orig = preprocessed_img_orig / 255.0
preprocessed_img_orig = tf.clip_by_value(preprocessed_img_orig, 0.0, 1.0).numpy()

# Generate the attention heatmaps.
attentions = attention_heatmap(attention_score_dict, preprocessed_img_orig)

# Plot the maps.
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(13, 13))
img_count = 0

for i in range(3):
    for j in range(4):
        if img_count < len(attentions):
            axes[i, j].imshow(preprocessed_img_orig[0])
            axes[i, j].imshow(attentions[..., img_count], cmap="inferno", alpha=0.6)
            axes[i, j].title.set_text(f"Attention head: {img_count}")
            axes[i, j].axis("off")
            img_count += 1




#Visualizing the learned projection filters

projections = (
    vit_base_i21k_patch16_224.layers[1]
    .get_layer("projection")
    .get_layer("conv_projection")
    .kernel.numpy()
)
projection_dim = projections.shape[-1]
patch_h, patch_w, patch_channels = projections.shape[:-1]


scaled_projections = MinMaxScaler().fit_transform(
    projections.reshape(-1, projection_dim)
)


scaled_projections = scaled_projections.reshape(patch_h, patch_w, patch_channels, -1)

# Visualize the first 128 filters of the learned projections.
fig, axes = plt.subplots(nrows=8, ncols=16, figsize=(13, 8))
img_count = 0
limit = 128

for i in range(8):
    for j in range(16):
        if img_count < limit:
            axes[i, j].imshow(scaled_projections[..., img_count])
            axes[i, j].axis("off")
            img_count += 1

fig.tight_layout()

#Visualizing the positional emebddings
position_embeddings = vit_base_i21k_patch16_224.layers[1].positional_embedding.numpy()
position_embeddings = position_embeddings.squeeze()[1:, ...]

similarity = position_embeddings @ position_embeddings.T
plt.imshow(similarity, cmap="inferno")
plt.show()
