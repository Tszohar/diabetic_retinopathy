import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


class Analyzer:
    def __init__(self, csv_path):
        self.dataframe = pd.read_csv(csv_path)
        self.predicted_labels = self.dataframe['predicted_label']
        self.converted_labels = self.dataframe['converted_label']
        self.diagnosis = self.dataframe['diagnosis']
        self.scores = self.dataframe['scores']

    def accuracy(self, predicted_labels, correct_labels):
        raise NotImplementedError

    def accuracy_per_class(self, num_of_classes, predicted_labels, correct_labels):
        class_accuracy = np.zeros(num_of_classes).astype(float)
        for idx in range(num_of_classes):
            class_accuracy[idx] = self.accuracy(predicted_labels=predicted_labels[correct_labels == idx],
                                                correct_labels=correct_labels[correct_labels == idx])
            print('Class: {}, Accuracy: {}'.format(idx, class_accuracy[idx]))
        return class_accuracy

    def accuracy2(self, predicted_labels, correct_labels):
        num_correct_labels = accuracy_score(y_true=predicted_labels, y_pred=correct_labels, normalize=False)
        accuracy = (num_correct_labels / len(correct_labels))
        print('Accuracy: {}'.format(accuracy))
        return accuracy

    def mislabeled_images(self, predicted_labels, correct_labels):
        return predicted_labels[predicted_labels != correct_labels]

    def conf_matrix(self, predicted_labels, correct_labels):
        conf_m = confusion_matrix(y_true=correct_labels, y_pred=predicted_labels)
        return conf_m

    def recall_by_class(self,num_of_classes, predicted_labels, correct_labels):
        conf_m = self.conf_matrix(predicted_labels=predicted_labels, correct_labels=correct_labels)
        recall = np.zeros(num_of_classes).astype(float)
        for idx in range(num_of_classes):
            recall[idx] = conf_m[idx, idx] / sum(conf_m[idx, :])
        return recall

    def precision_by_class(self, num_of_classes, predicted_labels, correct_labels):
        conf_m = self.conf_matrix(predicted_labels=predicted_labels, correct_labels=correct_labels)
        precision = np.zeros(num_of_classes).astype(float)
        for idx in range(num_of_classes):
            ###### Ask Guy: how to avoide dividing by zero
            precision[idx] = conf_m[idx, idx] / sum(conf_m[:, idx])
        return precision

    def f1_by_class(self, num_of_classes, predicted_labels, correct_labels):
        f1 = np.zeros(num_of_classes).astype(float)
        recall = self.recall_by_class(num_of_classes=num_of_classes, predicted_labels=predicted_labels,
                                      correct_labels=correct_labels)
        percision = self.recall_by_class(num_of_classes=num_of_classes, predicted_labels=predicted_labels,
                                         correct_labels=correct_labels)
        for idx in range(num_of_classes):
            f1[idx] = 2 * (percision[idx] * recall[idx]) / (percision[idx] + recall[idx])
        return f1

    def statistics(self, num_of_classes, predicted_labels, correct_labels, output_path):

        accuracy = self.accuracy(predicted_labels=predicted_labels, correct_labels=correct_labels)
        accuracy_per_class = self.accuracy_per_class(num_of_classes=num_of_classes,
                                                     predicted_labels=predicted_labels,
                                                     correct_labels=correct_labels)
        conf_m = self.conf_matrix(predicted_labels=predicted_labels, correct_labels=correct_labels)

        recall = self.recall_by_class(predicted_labels=predicted_labels, correct_labels=correct_labels,
                                      num_of_classes=num_of_classes)

        percision = self.precision_by_class(predicted_labels=predicted_labels, correct_labels=correct_labels,
                                            num_of_classes=num_of_classes)

        # f1 = self.f1_by_class(predicted_labels=predicted_labels, correct_labels=correct_labels,
        #                       num_of_classes=num_of_classes)

        statistics_df = pd.DataFrame({'total_accuracy': np.array(accuracy),
                                      'class#': range(num_of_classes),
                                      'accuracy_per_class': (accuracy_per_class),
                                      'recall': recall,
                                      'percision': percision,
                                   #   'F1': f1,
                                      })
        self.save_csv(statistics_df, output_path)

    def save_csv(self, dataframe, csv_path):
        dataframe.to_csv(csv_path)


class RegressorAnalyzer(Analyzer):
    def __init__(self, csv_path):
        super().__init__(csv_path)

    def accuracy(self, predicted_labels, correct_labels):
        num_correct_labels = accuracy_score(y_true=predicted_labels.round(), y_pred=correct_labels, normalize=False)
        accuracy = (num_correct_labels / len(correct_labels))
        print('Accuracy: {}'.format(accuracy))
        return accuracy


class BinaryAnalyzer(Analyzer):
    def __init__(self, csv_path):
        super().__init__(csv_path)

    def accuracy(self, predicted_labels, correct_labels):
        return self.accuracy2(predicted_labels=predicted_labels,correct_labels=correct_labels)


class MultiBinaryAnalyzer(Analyzer):
    def __init__(self, csv_path):
        super().__init__(csv_path)

    def accuracy(self, predicted_labels, correct_labels):
        return self.accuracy2(predicted_labels=predicted_labels,correct_labels=correct_labels)


class MultiClassAnalyzer(Analyzer):
    def __init__(self, csv_path):
        super().__init__(csv_path)

    def accuracy(self, predicted_labels, correct_labels):
        return self.accuracy2(predicted_labels=predicted_labels,correct_labels=correct_labels)



