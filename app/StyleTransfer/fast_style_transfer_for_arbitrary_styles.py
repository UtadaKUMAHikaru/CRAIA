# 图片可以来自http url，也可以来自path
# 写个默认的情况？
# load_image中要不要crop

import functools
import os, io

# from matplotlib import gridspec
# import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image

content_paths = dict(
    golden_gate_bridge = 'content_images/Golden_Gate_Bridge_from_Battery_Spencer.jpeg',
    sea_turtle='content_images/Green_Sea_Turtle_grazing_seagrass.jpeg',
    tuebingen='content_images/Tuebingen_Neckarfront.jpeg',
    grace_hopper='content_images/grace_hopper.jpeg',
    )
style_paths = dict(
    kanagawa_great_wave='style_images/The_Great_Wave_off_Kanagawa.jpeg',
    kandinsky_composition_7='style_images/Vassily_Kandinsky,_1913_-_Composition_7.jpeg',
    hubble_pillars_of_creation='style_images/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpeg',
    van_gogh_starry_night='style_images/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpeg',
    turner_nantes='style_images/JMW_Turner_-_Nantes_from_the_Ile_Feydeau.jpeg',
    munch_scream='style_images/Edvard_Munch,_1893,_The_Scream,_oil,_tempera_and_pastel_on_cardboard,_91_x_73_cm,_National_Gallery_of_Norway.jpeg',
    picasso_demoiselles_avignon="style_images/Les_Demoiselles_d'Avignon.jpeg",
    picasso_violin='style_images/Pablo_Picasso,_1911-12,_Violon_(Violin),_oil_on_canvas,_Kröller-Müller_Museum,_Otterlo,_Netherlands.jpeg',
    picasso_bottle_of_rum='style_images/Pablo_Picasso,_1911,_Still_Life_with_a_Bottle_of_Rum,_oil_on_canvas,_61.3_x_50.5_cm,_Metropolitan_Museum_of_Art,_New_York.jpg',
    fire='style_images/Large_bonfire.jpeg',
    derkovits_woman_head='style_images/Derkovits_Gyula_Woman_head_1922.jpeg',
    amadeo_style_life='style_images/Untitled_(Still_life)_(1913)_-_Amadeo_Souza-Cardoso_(1887-1918)_(17385824283).jpeg',
    derkovtis_talig='style_images/Derkovits_Gyula_Taligás_1920.jpeg',
    amadeo_cardoso='style_images/Amadeo_de_Souza-Cardoso,_1915_-_Landscape_with_black_figure.jpeg'
  )

content_urls = dict(
  golden_gate_bridge='https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/Golden_Gate_Bridge_from_Battery_Spencer.jpg/640px-Golden_Gate_Bridge_from_Battery_Spencer.jpg',
  sea_turtle='https://upload.wikimedia.org/wikipedia/commons/d/d7/Green_Sea_Turtle_grazing_seagrass.jpg',
  tuebingen='https://upload.wikimedia.org/wikipedia/commons/0/00/Tuebingen_Neckarfront.jpg',
  grace_hopper='https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg',
  )
style_urls = dict(
  kanagawa_great_wave='https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg',
  kandinsky_composition_7='https://upload.wikimedia.org/wikipedia/commons/b/b4/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg',
  hubble_pillars_of_creation='https://upload.wikimedia.org/wikipedia/commons/6/68/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg',
  van_gogh_starry_night='https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg',
  turner_nantes='https://upload.wikimedia.org/wikipedia/commons/b/b7/JMW_Turner_-_Nantes_from_the_Ile_Feydeau.jpg',
  munch_scream='https://upload.wikimedia.org/wikipedia/commons/c/c5/Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg',
  picasso_demoiselles_avignon='https://upload.wikimedia.org/wikipedia/en/4/4c/Les_Demoiselles_d%27Avignon.jpg',
  picasso_violin='https://upload.wikimedia.org/wikipedia/en/3/3c/Pablo_Picasso%2C_1911-12%2C_Violon_%28Violin%29%2C_oil_on_canvas%2C_Kr%C3%B6ller-M%C3%BCller_Museum%2C_Otterlo%2C_Netherlands.jpg',
  picasso_bottle_of_rum='https://upload.wikimedia.org/wikipedia/en/7/7f/Pablo_Picasso%2C_1911%2C_Still_Life_with_a_Bottle_of_Rum%2C_oil_on_canvas%2C_61.3_x_50.5_cm%2C_Metropolitan_Museum_of_Art%2C_New_York.jpg',
  fire='https://upload.wikimedia.org/wikipedia/commons/3/36/Large_bonfire.jpg',
  derkovits_woman_head='https://upload.wikimedia.org/wikipedia/commons/0/0d/Derkovits_Gyula_Woman_head_1922.jpg',
  amadeo_style_life='https://upload.wikimedia.org/wikipedia/commons/8/8e/Untitled_%28Still_life%29_%281913%29_-_Amadeo_Souza-Cardoso_%281887-1918%29_%2817385824283%29.jpg',
  derkovtis_talig='https://upload.wikimedia.org/wikipedia/commons/3/37/Derkovits_Gyula_Talig%C3%A1s_1920.jpg',
  amadeo_cardoso='https://upload.wikimedia.org/wikipedia/commons/7/7d/Amadeo_de_Souza-Cardoso%2C_1915_-_Landscape_with_black_figure.jpg'
)


def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

@functools.lru_cache(maxsize=None) #调了大小
def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
  """Loads and preprocesses images."""
  # Cache image file locally.
  image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = tf.io.decode_image(
      tf.io.read_file(image_path),
      channels=3, dtype=tf.float32)[tf.newaxis, ...]
  # img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

@functools.lru_cache(maxsize=None)
def load_image_path(image_path, image_size=(256, 256), preserve_aspect_ratio=True):
  """Loads and preprocesses images."""
  # Cache image file locally.
  # image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = tf.io.decode_image(
      tf.io.read_file(image_path),
      channels=3, dtype=tf.float32)[tf.newaxis, ...]
  # img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img

# def show_n(images, titles=('',)):
#   n = len(images)
#   image_sizes = [image.shape[1] for image in images]
#   w = (image_sizes[0] * 6) // 320
#   plt.figure(figsize=(w * n, w))
#   gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
#   for i in range(n):
#     plt.subplot(gs[i])
#     plt.imshow(images[i][0], aspect='equal')
#     plt.axis('off')
#     plt.title(titles[i] if len(titles) > i else '')
#   plt.show()

# output_image_size = 384  # @param {type:"integer"}
# The content image size can be arbitrary.
# content_img_size = (output_image_size, output_image_size)
# style_img_size = (256, 256)  # Recommended to keep it at 256.

print("TF Version: ", tf.__version__)
print("TF Hub version: ", hub.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
if tf.test.is_gpu_available():  # returns True if TensorFlow is using GPU
  # 有GPU时
  print("GPU available: ", tf.config.list_physical_devices('GPU'))
else:
  # 无GPU时
  print("GPU available: ", tf.config.experimental.list_physical_devices(device_type=None))

#模型下载到本地

hub_handle = 'StyleTransfer/magenta_arbitrary-image-stylization-v1-256_2'
# hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

def style_transfer_simple(content_image_url, style_image_url, parameters):
  

  content_image_size = 384
  style_image_size = 256
  # content_images = {k: load_image(v, (content_image_size, content_image_size)) for k, v in content_urls.items()}
  # style_images = {k: load_image(v, (style_image_size, style_image_size)) for k, v in style_urls.items()}
  
  # content_images = {k: load_image_path(v, (content_image_size, content_image_size)) for k, v in content_paths.items()}
  # style_images = {k: load_image_path(v, (style_image_size, style_image_size)) for k, v in style_paths.items()}
  # style_images = {k: tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME') for k, style_image in style_images.items()}

  # 可调参数
  #@title Specify the main content image and the style you want to use.  { display-mode: "form" }

  content_name = 'tuebingen'  # @param ['sea_turtle', 'tuebingen', 'grace_hopper']
  style_name = 'munch_scream'  # @param ['kanagawa_great_wave', 'kandinsky_composition_7', 'hubble_pillars_of_creation', 'van_gogh_starry_night', 'turner_nantes', 'munch_scream', 'picasso_demoiselles_avignon', 'picasso_violin', 'picasso_bottle_of_rum', 'fire', 'derkovits_woman_head', 'amadeo_style_life', 'derkovtis_talig', 'amadeo_cardoso']

  content_image = tf.constant(load_image(content_image_url, (content_image_size, content_image_size)))
  style_image = tf.constant(load_image(style_image_url, (style_image_size, style_image_size)))
  # content_image = tf.constant(content_images[content_name])
  # style_image = tf.constant(style_images[style_name])

  stylized_image = hub_module(content_image, style_image)[0]

  # tf.io.encode_jpg(stylized_image,compression=-1,name='output.jpg')
  # print(stylized_image.)

  image = Image.fromarray(np.uint8(stylized_image[0].numpy()*255))
  # image = Image.fromarray(stylized_image[0].numpy())
  # image = Image.fromarray(stylized_image[0].numpy(), mode="RGB")

  # image.save("StyleTransfer/output/output.png")
  
  # show_n([content_images[content_name], style_images[style_name], stylized_image],
  #       titles=['Original content image', 'Style image', 'Stylized image'])

  # https://stackoverflow.com/questions/55873174/how-do-i-return-an-image-in-fastapi
    # https://fastapi.tiangolo.com/advanced/custom-response/#htmlresponse
  # image_bytes: bytes = generate_cat_picture()
  # media_type here sets the media type of the actual response sent to the client.
  

  img_byte_arr = io.BytesIO()
  image.save(img_byte_arr, format='PNG')
  img_byte_arr = img_byte_arr.getvalue()
  return img_byte_arr
  # return stylized_image

if __name__ == '__main__':
  style_transfer()
