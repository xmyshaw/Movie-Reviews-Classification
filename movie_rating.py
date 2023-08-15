#!/usr/bin/python
import sqlite3
import jieba
from sqlite3 import Error
from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, LSTM
import matplotlib.pyplot as plt

# Read .db file and store rating and content in lists
# Create a database connection
def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
 
    return None

# Creat database path
database = "data/douban.db"
# Connect database
conn = create_connection(database)

with conn:
    """
    Query all rows in the comment table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT RATING, CONTENT FROM comment WHERE RATING IN (1,2,3,4,5)")
    rows = cur.fetchall()#return as a list

    # Store the rating and content from the .db file to lists
    rating = []
    content = []
    for row in rows:
        rating.append(row[0])
        content.append(row[1]) 

print("length of the list rating: ", len(rating))
print("length of the list content: ", len(content))
print("print the 10th element: ", rating[10],content[10])



# Create stopwords list
def get_custom_stopwords(stop_words_file):
    # Open the file
    with open(stop_words_file) as f:
        stopwords = f.read()
    # Split with new line
    stopwords_list = stopwords.split('\n')
    # Put into the list
    custom_stopwords_list = [i for i in stopwords_list]
    return custom_stopwords_list

stop_words_file = "data/stopwordsHIT.txt"
stopwords = get_custom_stopwords(stop_words_file)

print(stopwords[-10:])

# Segmentation
seg_content = []


for i in content:
    outstr = ""
    #segment each comment
    seg = jieba.cut(i, cut_all=False, HMM=True)
    #iterate the comment and check if it contains stopwords
    for word in seg:
        #if the word is not a stopword, attach it to string outstr
        if word not in stopwords:
            if word != '\t':
                outstr += word
                outstr += " "
    #append all strings (segmented comments) in a list            
    seg_content.append(outstr)

print("length of seg_content: " , len(seg_content))
print("print the seg_content[:5]:" , seg_content[:5])





#tokenization

#allow at most 100 words in each comment
max_words = 10000

#only use the most frequent 10000 words
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(seg_content)
sequences = tokenizer.texts_to_sequences(seg_content)

print("type of sequences: " , type(sequences))
print("print sequences[:1]" , sequences[:1])

print("length of sequences:")
for sequence in sequences[:100]:
    print(len(sequence))


maxlen = 30
#make all sequences the same length
data = pad_sequences(sequences, maxlen=maxlen)

print(data)

#save words and their sequences in a dictionary
word_index = tokenizer.word_index
#print(word_index)



#labels to one hot
encoder = LabelBinarizer()
labels = encoder.fit_transform(rating)
print("print labels:" , labels)



#shuffle data and labels
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]



#split data into trian, test, validation

X_train_data, X_test, y_train_data, y_test = train_test_split(data, labels, test_size=0.3, random_state = 42)

training_samples = int(len(X_train_data) * .8)
validation_samples = len(X_train_data) - training_samples

X_train = X_train_data[:training_samples]
y_train = y_train_data[:training_samples]
X_valid = X_train_data[training_samples: training_samples + validation_samples]
y_valid = y_train_data[training_samples: training_samples + validation_samples]

#print(X_train)



#word embedding
zh_model = KeyedVectors.load_word2vec_format('data/zh.vec')

#print a vector
print(zh_model.vectors[0])
#print the first 5 words
print(list(iter(zh_model.vocab))[:5])
#print the total dimensions
print(len(zh_model[next(iter(zh_model.vocab))]))


embedding_dim = len(zh_model[next(iter(zh_model.vocab))])
#build a random matrix
embedding_matrix = np.random.rand(max_words, embedding_dim)
print(embedding_matrix.shape)

#a formula to turn the numbers to the range between -1 and 1
embedding_matrix = (embedding_matrix - 0.5) * 2

for word, i in word_index.items():
    if i < max_words:
        try:
            #if the word is in the zh.vec, change the randomly generated vector to the vector retrieved from zh.vec
            embedding_vector = zh_model.get_vector(word)
            embedding_matrix[i] = embedding_vector
        #if the word is not in the zh.vec, still use our randomly generated vector
        except:
            pass


units = 32

def build_model():
    model = Sequential()
    model.add(Embedding(max_words, embedding_dim))
    model.add(LSTM(units))
    model.add(Dense(5, activation='softmax'))
    model.summary()

    #add our embedding matrix to the embedding laye
    model.layers[0].set_weights([embedding_matrix])
    model.layers[0].trainable = False

    model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['acc'])

    return model

model = build_model()

history = model.fit(X_train, y_train,
                    epochs=4,
                    batch_size=64,
                    validation_data=(X_valid, y_valid))
model.save("result/mymodel.h5")


#visualization


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

#draw accuracy graph
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

#draw loss graph
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


#evaluate on test data
results = model.evaluate(X_test,y_test)
print(results)
