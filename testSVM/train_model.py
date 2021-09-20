from sklearn import svm
import joblib


def train_model(train_data, train_label):
    model = svm.SVC(decision_function_shape='ovo')
    model.fit(train_data, train_label)
    joblib.dump(model, "model/" + "model" + '.pickle')

#
# if __name__ == '__main__':
#     train_model([], 'a')
