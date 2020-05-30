import pickle
import sklearn
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

#load embeddings.pickle
print("[INFO] loading face embeddings...")
data = pickle.loads(open('./embeddings/embeddings.pickle','rb').read())

#encode labels(names)
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data['names'])

#train recognizer(SVM)
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel='linear', probability=True)
recognizer.fit(data['embeddings'],labels)

#save recognizer(SVM)
with open('./recognizer/recognizer.pickle', 'wb') as f:
    f.write(pickle.dumps(recognizer))

#save label encodings
with open('./label_encoder/le.pickle', 'wb') as f:
    f.write(pickle.dumps(le))