import os 
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction, modified_precision

cnn_model = VGG16()
feature_extractor = Model(inputs=cnn_model.inputs, outputs=cnn_model.layers[-2].output)
print(feature_extractor.summary())

IMAGE_DIR = 'ImageCaptioning\Flicker8k_Dataset'

image_features = {}

with tf.device('/job:localhost/replica:0/task:0/device:GPU:1'):
    for imageName in os.listdir(IMAGE_DIR):
        imagePath = IMAGE_DIR + '/' + imageName
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)
        feature = feature_extractor.predict(image, verbose=0)
        imageID = imageName.split('.')[0]
        image_features[imageID] = feature

SAVE_DIR = r'C:\Users\mosai\Desktop\internships\CodSoft internship\CODSOFT\ImageCaptioning'
pickle.dump(image_features, open(os.path.join(SAVE_DIR, 'image_features.pkl'), 'wb'))
with open(os.path.join(SAVE_DIR, 'image_features.pkl'), 'rb') as f:
    image_features = pickle.load(f)

with open(('ImageCaptioning/captions.txt'), 'r') as f:
    next(f)
    caption_text = f.read()  
caption_dict = {}

for line in (caption_text.split('\n')):
    parts = line.split(',')
    if len(line) < 2:
        continue
    img_id, caption = parts[0], parts[1:]
    img_id = img_id.split('.')[0]
    caption = " ".join(caption)
    if img_id not in caption_dict:
        caption_dict[img_id] = []
    caption_dict[img_id].append(caption)

len(caption_dict)

def preprocess_captions(caption_dict):
    for key, captions in caption_dict.items():
        for i in range(len(captions)):
            caption = captions[i]
            caption = caption.lower()
            caption = caption.replace('[^A-Za-z]', '')
            caption = caption.replace('\s+', ' ')
            caption = 'startseq ' + " ".join([word for word in caption.split() if len(word)>1]) + ' endseq'
            captions[i] = caption

preprocess_captions(caption_dict)

all_captions = []
for key in caption_dict:
    for caption in caption_dict[key]:
        all_captions.append(caption)   

len(all_captions)
all_captions[:10]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
vocab_size
max_length = max(len(caption.split()) for caption in all_captions)
max_length
img_ids = list(caption_dict.keys())
split_index = int(len(img_ids) * 0.90)
train_ids = img_ids[:split_index]
test_ids = img_ids[split_index:]

def data_generator(data_keys, caption_dict, image_features, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = list(), list(), list()
    n = 0
    while 1:
        for key in data_keys:
            n += 1
            captions = caption_dict[key]
            for caption in captions:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    
                    X1.append(image_features[key][0])
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                X1, X2, y = np.array(X1), np.array(X2), np.array(y)
                yield {"image": X1, "text": X2}, y
                X1, X2, y = list(), list(), list()
                n = 0

image_input = Input(shape=(4096,), name="image")
img_features = Dropout(0.4)(image_input)
img_features = Dense(256, activation='relu')(img_features)

text_input = Input(shape=(max_length,), name="text")
text_features = Embedding(vocab_size, 256, mask_zero=True)(text_input)
text_features = Dropout(0.4)(text_features)
text_features = LSTM(256)(text_features)

decoder = add([img_features, text_features])
decoder = Dense(256, activation='relu')(decoder)
output = Dense(vocab_size, activation='softmax')(decoder)

caption_model = Model(inputs=[image_input, text_input], outputs=output)
caption_model.compile(loss='categorical_crossentropy', optimizer='adam')
#plot_model(caption_model, show_shapes=True)

epochs = 20
batch_size = 32
steps_per_epoch = len(train_ids) // batch_size

with tf.device('/job:localhost/replica:0/task:0/device:GPU:1'):
    for i in range(epochs):
        generator = data_generator(train_ids, caption_dict, image_features, tokenizer, max_length, vocab_size, batch_size)
        caption_model.fit(generator, epochs=1, steps_per_epoch=steps_per_epoch, verbose=1)

caption_model.save(SAVE_DIR+'/caption_model.h5')

def index_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_caption(model, image, tokenizer, max_length):
    caption = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([caption])[0]
        sequence = pad_sequences([sequence], max_length)
        pred = model.predict([image, sequence], verbose=0)
        pred = np.argmax(pred)
        word = index_to_word(pred, tokenizer)
        if word is None:
            break
        caption += " " + word
        if word == 'endseq':
            break
      
    return caption

actual_captions, predicted_captions = list(), list()

for key in test_ids:
    captions = caption_dict[key]
    predicted = generate_caption(caption_model, image_features[key], tokenizer, max_length) 
    actual = [caption.split() for caption in captions]
    predicted = predicted.split()
    actual_captions.append(actual)
    predicted_captions.append(predicted)
    

def corpus_bleu_fixed(list_of_references, hypotheses, weights):
    p_n = (modified_precision(list_of_references, hyp, i + 1) for i, hyp in enumerate(hypotheses))
    s = SmoothingFunction().method1
    return corpus_bleu(list_of_references, hypotheses, weights, smoothing_function=s)

print("BLEU-1: %f" % corpus_bleu(actual_captions, predicted_captions, weights=(1.0, 0, 0, 0)))
print("BLEU-2: %f" % corpus_bleu(actual_captions, predicted_captions, weights=(0.5, 0.5, 0, 0)))

from PIL import Image
import matplotlib.pyplot as plt
def display_caption(image_name):
    img_id = image_name.split('.')[0]
    img_path = os.path.join(IMAGE_DIR, "Images", image_name)
    image = Image.open(img_path)
    captions = caption_dict[img_id]
    print('---------------------Actual---------------------')
    for caption in captions:
        print(caption)
    predicted = generate_caption(caption_model, image_features[img_id], tokenizer, max_length)
    print('--------------------Predicted--------------------')
    print(predicted)
    plt.imshow(image)