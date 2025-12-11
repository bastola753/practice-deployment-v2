import pickle

cv = pickle.load(open("models/cv.pkl","rb"))
clf = pickle.load(open("models/clf.pkl","rb"))

def make_prediction(email):
    tokenized_email = cv.transform([email])
    prediction = clf.predict(tokenized_email)
    return 1 if prediction[0]==1 else -1