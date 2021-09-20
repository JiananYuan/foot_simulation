from sklearn import svm
import joblib


def train_model(train_data, name):
    print("Training " + name + "\'s model")
    model = svm.OneClassSVM(nu=0.9, kernel='rbf', gamma='auto')
    model.fit(train_data)
    joblib.dump(model, "model/" + name + '.pickle')
    # model = svm.SVC(decision_function_shape='ovo')
    # train_label = [1]*50 + [2]*50 + [3]*50 + [4]*50 + [5]*50
    # model.fit(train_data, train_label)
    # joblib.dump(model, "model/" + name + '.pickle')

#
# if __name__ == '__main__':
#     train_model([], 'a')
