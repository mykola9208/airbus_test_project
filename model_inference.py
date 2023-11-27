import pandas as pd
import os
import numpy as np
from imageio.v2 import imread
from skimage.transform import resize
# щоб не повторювати параметри, грузимо їх з іншого файлу та грузимо модель
from model_training import model, image_shape, IMG_WIDTH, IMG_HEIGHT, TARGET_WIDTH, TARGET_HEIGHT, data_dir


FAST_PREDICTION = False  # використовується для більш швидкої перевірки роботи програми
FAST_PREDICTION_SIZE = 5000
sub_df = pd.read_csv(os.path.join(data_dir, 'sample_submission_v2.csv'))

no_mask = np.zeros(image_shape[0]*image_shape[1], dtype=np.uint8)

if FAST_PREDICTION:
    sub_df = sub_df.sample(n=FAST_PREDICTION_SIZE).reset_index().drop(columns=["index"])  # after reset index dataframe will have one more column call index

#завантажуємо результати тренування моделі
model.load_weights('../airbus_test_project/seg_model_weights.best.hdf5')

# описуємо функцію енкодера для маски, щоб записати потім у файл

def rle_encode(img):
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

#описуємо функції завантаження картинок
def get_test_image(image_name):
    img = imread(os.path.join(data_dir, 'test_v2/') + image_name)
    img = resize(img, (TARGET_WIDTH, TARGET_HEIGHT), mode='constant', preserve_range=True)
    return img


def create_test_generator():
    while True:
        for k, ix in sub_df.groupby(np.arange(sub_df.shape[0])):
            imgs = []
            for indx, rows in ix.iterrows():
                original_img = get_test_image(rows.ImageId) / 255.0
                imgs.append(original_img)

            imgs = np.array(imgs)
            yield imgs


test_generator = create_test_generator()

test_steps = np.ceil(float(sub_df.shape[0])).astype(int)
predict_mask = model.predict(test_generator, steps=test_steps)

#отримаємо предікт
for index, row in sub_df.iterrows():
    predict = predict_mask[index]
    resized_predict = resize(predict, (IMG_WIDTH, IMG_HEIGHT)) * 255
    mask = resized_predict > 0.5
    sub_df.at[index, 'EncodedPixels'] = rle_encode(mask)

# зберігаємо у необхідний файл, але я зберігаю у інший файл, щоб не змінювати кількість рядків у файлі sample_submission_v2.csv
sub_df.to_csv(os.path.join(data_dir, 'sample_submission_result.csv'), index=False)