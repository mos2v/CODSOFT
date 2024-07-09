from os import listdir
from pickle import dump, load
import tensorflow as tf
from keras.src.applications.vgg16 import VGG16
from keras._tf_keras.keras.preprocessing.image import load_img
from keras._tf_keras.keras.preprocessing.image import img_to_array
from keras.src.applications.vgg16 import preprocess_input
from keras.src.models import Model
import string
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences
from keras.src.utils import to_categorical
from keras.src.utils import plot_model
from keras.src.layers import Input
from keras.src.layers import Dense
from keras.src.layers import LSTM
from keras.src.layers import Embedding
from keras.src.layers import Dropout
from keras.src.layers.merging import add
from keras.src.callbacks import ModelCheckpoint
import numpy as np
from nltk.translate.bleu_score import corpus_bleu
from keras.src.saving import load_model


def FeaturesExtract(dir):
    model = VGG16()

    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

    print(model.summary())

    features = dict()

    for name in listdir(dir):
        filename = dir + '/' + name

        image = load_img(filename, target_size=(224, 224))
        image = img_to_array(image)

        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

        image = preprocess_input(image)

        feature = model.predict(image, verbose=0)

        imageID = name.split('.')[0]

        features[imageID] = feature

        print('>%s' % name)

    return features

directory = 'ImageCaptioning\Flicker8k_Dataset'
features = FeaturesExtract(directory)
print('Extracted Features:  %d' % len(features))

dump(features, open('dumped/features.pkl', 'wb'))

def loadDocument(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def loadDescriptions(document):
    mapping = dict()

    for line in document.split('\n'):
        tokens = line.split()
        if len(line) < 2:
            continue
        imageID, imageDescription = tokens[0], tokens[1:]
        imageID = imageID.split('.')[0]
        imageDescription = ' '.join(imageDescription)

        if imageID not in mapping:
            mapping[imageID] = list()
        mapping[imageID].append(imageDescription)

    return mapping

def DescriptionsClean(descrip):
    table = str.maketrans('','', string.punctuation)
    for key, descriptionList in descrip.items():
        for i in range(len(descriptionList)):
            description = descriptionList[i]

            description = description.split()

            description = [word.lower() for word in description]

            description = [word.translate(table) for word in description]

            description = [word for word in description if len(word) > 1]

            description = [word for word in description if word.isalpha()]

            descriptionList[i] = ' '.join(description)


def Vocab(descrip):
    description = set()

    for key in descrip.keys():
        [description.update(a.split()) for a in descrip[key]]

    return description


def SaveDescription(descrip, filename):
    lines = list()
    for key, descriptionList in descrip.items():
        for description in descriptionList:
            lines.append(key +'' + description)

    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


filename = 'ImageCaptioning\Flickr8k.token.txt'  

doc = loadDocument(filename)

descriptions = loadDescriptions(doc)

print('descriptions Loaded: %d ' % len(descriptions))

DescriptionsClean(descriptions)

vocab = Vocab(descriptions)

print('Vocabulary Size: %d' % len(vocab))

SaveDescription(descriptions, 'ImageCaptioning\descriptions.txt')

def loadSet(filename):
    doc = loadDocument(filename)
    dataset = list()
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        ID = line.split('.')[0]
        dataset.append(ID)
    return set(dataset)

def loadCleanDescriptions(file, dataset):
    doc = loadDocument(file)
    descriptions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        imageID, imageDescription = tokens[0], tokens[1:]
        if imageID in dataset:
            if imageID not in descriptions:
                descriptions[imageID] = list()
            description = 'startseq ' + ' '.join(imageDescription) + ' endseq'

            descriptions[imageID].append(description)
    return descriptions

def loadPhotoFeatures(file, dataset):
    features = load(open(file, 'rb'))
    features = {k: features[k] for k in dataset}
    return features

filename1 = 'ImageCaptioning\Flickr_8k.trainImages.txt'
train = loadSet(filename1)

print('Dataset: %d' % len(train))

trainDescriptions = loadCleanDescriptions('ImageCaptioning\descriptions.txt', train)

print('Descriptions: %d' % len(trainDescriptions))

trainFeatures = loadPhotoFeatures('dumped/features.pkl', train)
print('Photos: train=%d ' % len(trainFeatures))


def Lines(descrip):
    descriptions = list()
    for k in descrip.keys():
        [descriptions(d) for d in descrip[k]]
    return descriptions    


def tokenize(descriptions):
    lines = Lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

tokenizer = tokenize(trainDescriptions)
vocabularySize = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocabularySize)

def length(descrip):
    lines = Lines(descrip)
    return max(len(d.split()) for d in lines)

def Sequence(tokenizer, maxLength, descriptionList, img):
    x1, x2, y = list(), list(), list()
    for d in descriptionList:
        sequence = tokenizer.texts_to_sequences([d])[0]

        for i in range(1, len(sequence)):
            inputSequence, outputSequence = sequence[:i], sequence[i]

            inputSequence = pad_sequences([inputSequence], maxlen=maxLength)[0]
            outputSequence = to_categorical([outputSequence], num_classes=vocabularySize)[0]

            x1.append(img)
            x2.append(inputSequence)
            y.append(outputSequence)
    return np.array(x1), np.array(x2), np.array(y)       

def CaptionModel(VocabularySize, maxLength):
    inputImage = Input(shape=(4096,))
    reg = Dropout(0.5)(inputImage)
    hiddenImage = Dense(256, activation='relu')(reg)

    inputText = Input(shape=(maxLength,))
    param1 = Embedding(vocabularySize, 256, mask_zero=True)(inputText)
    param2 = Dropout(0.5)(param1)
    hiddenText = LSTM(256)(param2)

    decoder1 = add([hiddenImage, hiddenText])
    decoder2 = Dense(256, activation='relu')(decoder1)
    output = Dense(vocabularySize, activation='softmax')(decoder2)

    model = Model(inputs=[inputImage, inputText], outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    print(model.summary())
    return model

def dataGene(descrip, imgs, tokenizer, maxLength):
    while 1:
        for k, descriptionList in descrip.items():
            img = imgs[k][0]
            inputImage, InputText, outputWords = Sequence(tokenizer, maxLength, descriptionList, img)
            yield [[inputImage, InputText], outputWords]

maxLength = length(trainDescriptions)
print('Description Length: %d' % maxLength)

model = CaptionModel(vocabularySize, maxLength)
epochs = 20
steps = len(trainDescriptions)

for i in range(epochs):

    generator = dataGene(trainDescriptions, trainFeatures, tokenizer, maxLength)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    model.save('dumped/model_' + str(i) + '.h5')


def WordID(int, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == int:
            return word
    return None

def DescriptionGenerator(model, tokenizer, img, maxLength):
    inputText = 'startseq'

    for i in range(maxLength):
        sequence = tokenizer.texts_to_sequences([inputText])[0]
        sequence = pad_sequences([sequence], maxlen=maxLength)
        yhat = model.predict([img, sequence], verbose=0)
        yhat = np.argmax(yhat)

        word = WordID(yhat, tokenizer)

        if word is None:
            break
        inputText += ' ' + word
        if word == 'endseq':
            break
    return inputText    


def modelEval(model, descriptions, photos, tokenizer, maxLength):
    Actual, Predicted = list(), list()

    for k, descriptionList in descriptions.items():
        yhat = DescriptionGenerator(model, tokenizer, photos[k], maxLength)

        ref = [d.split() for d in descriptionList]
        Actual.append(ref)
        Predicted.append(yhat.split())
    
    
    print('BLEU-1: %f' % corpus_bleu(Actual, Predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(Actual, Predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(Actual, Predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(Actual, Predicted, weights=(0.25, 0.25, 0.25, 0.25)))
   



TestFile = 'ImageCaptioning\Flickr_8k.testImages.txt'

test = loadSet(TestFile)

print('Test Dataset: %d' % len(test))

testDescriptions = loadCleanDescriptions('ImageCaptioning\descriptions.txt', test)

print('Test Descriptions: test=%d' % len(testDescriptions))

testFeatures = loadPhotoFeatures('dumped/features.pkl', test)

print('Test Photos: test=%d' % len(testFeatures))

BestModel = 'dumped/model_18.h5'
model1 = load_model(BestModel)

modelEval(model1, testDescriptions, testFeatures, tokenizer, maxLength)