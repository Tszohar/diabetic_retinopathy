import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score


class Analyzer:
    def __init__(self, csv_path):
        self.dataframe = pd.read_csv(csv_path)

    def accuracy(self, predicted_labels, correct_labels):
        raise NotImplementedError

    def accuracy2(self, predicted_labels, correct_labels):
        accuracy = accuracy_score(y_true=predicted_labels, y_pred=correct_labels)
        return accuracy

    def kappa(self, predicted_labels, correct_labels):
        kappa_score = cohen_kappa_score(y1=predicted_labels, y2=correct_labels)
        return kappa_score

    def mislabeled_images(self, predicted_labels, correct_labels, output_path):
        mislabeled_idxs = np.where(predicted_labels != correct_labels)[0]
        mislabeled_df = pd.DataFrame({'id_code': self.dataframe['id_code'][mislabeled_idxs].squeeze(),
                                      'correct_label': correct_labels[mislabeled_idxs].squeeze(),
                                      'predicted_label': predicted_labels[mislabeled_idxs].squeeze(),
                                      'diagnosis': self.dataframe['diagnosis'][mislabeled_idxs].squeeze()
                                      })
        self.save_csv(mislabeled_df, output_path)
        return mislabeled_idxs

    def count_mislabeled_by_class(self, num_of_classes, output_path):
        mislabeled_idxs = self.mislabeled_images(predicted_labels=self.dataframe['predicted_label'],
                                                 correct_labels=self.dataframe['converted_label'],
                                                 output_path=output_path)
        count = np.zeros(num_of_classes)
        for idx in range(num_of_classes):
            count[idx] =len(self.dataframe['diagnosis'][mislabeled_idxs][self.dataframe['converted_label'] == idx])
        print('mislabeled images by class: ', count)
        return count

    def count_images_by_class(self, num_of_classes):
        count = np.zeros(num_of_classes)
        for idx in range(num_of_classes):
            count[idx] =len(self.dataframe['diagnosis'][self.dataframe['converted_label'] == idx])
        print('number of images by class: ', count)
        return count

    def conf_matrix(self, predicted_labels, correct_labels):
        conf_m = confusion_matrix(y_true=correct_labels, y_pred=predicted_labels)
        return conf_m

    def recall_by_class(self,num_of_classes, predicted_labels, correct_labels):
        conf_m = self.conf_matrix(predicted_labels=predicted_labels, correct_labels=correct_labels)
        recall = np.zeros(num_of_classes).astype(float)
        for idx in range(num_of_classes):
            recall[idx] = conf_m[idx, idx] / sum(conf_m[idx, :])
        print('Recall by class: ', recall)
        return recall

    def precision_by_class(self, num_of_classes, predicted_labels, correct_labels, eplison=1e-7):
        conf_m = self.conf_matrix(predicted_labels=predicted_labels, correct_labels=correct_labels)
        precision = np.zeros(num_of_classes).astype(float)
        for idx in range(num_of_classes):
            precision[idx] = conf_m[idx, idx] / (sum(conf_m[:, idx]) + eplison)
        print('Precision by class: ', precision)
        return precision

    def f1_by_class(self, num_of_classes, predicted_labels, correct_labels):
        f1 = np.zeros(num_of_classes).astype(float)
        recall = self.recall_by_class(num_of_classes=num_of_classes, predicted_labels=predicted_labels,
                                      correct_labels=correct_labels)
        precision = self.recall_by_class(num_of_classes=num_of_classes, predicted_labels=predicted_labels,
                                         correct_labels=correct_labels)
        for idx in range(num_of_classes):
            f1[idx] = 2 * (precision[idx] * recall[idx]) / (precision[idx] + recall[idx])
        return f1

    def statistics(self, num_of_classes, predicted_labels, correct_labels, output_path):

        accuracy = self.accuracy(predicted_labels=predicted_labels, correct_labels=correct_labels)
        print('total accuracy: {}'.format(accuracy))
        kappa = self.kappa(predicted_labels=predicted_labels, correct_labels=correct_labels)
        print('kappa score: {}'.format(kappa))

        conf_m = self.conf_matrix(predicted_labels=predicted_labels, correct_labels=correct_labels)
        print('conf matrix:')
        print(conf_m)

        recall = self.recall_by_class(predicted_labels=predicted_labels, correct_labels=correct_labels,
                                      num_of_classes=num_of_classes)

        percision = self.precision_by_class(predicted_labels=predicted_labels, correct_labels=correct_labels,
                                            num_of_classes=num_of_classes)

        # f1 = self.f1_by_class(predicted_labels=predicted_labels, correct_labels=correct_labels,
        #                       num_of_classes=num_of_classes)

        statistics_df = pd.DataFrame({'total_accuracy': np.array(accuracy),
                                      'class#': range(num_of_classes),
                                      'recall': recall,
                                      'percision': percision,
                                      'kappa': kappa,
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
        return accuracy


class BinaryAnalyzer(Analyzer):
    def __init__(self, csv_path):
        super().__init__(csv_path)

    def accuracy(self, predicted_labels, correct_labels):
        return self.accuracy2(predicted_labels=predicted_labels, correct_labels=correct_labels)


class MultiBinaryAnalyzer(Analyzer):
    def __init__(self, csv_path):
        super().__init__(csv_path)

    def accuracy(self, predicted_labels, correct_labels):
        return self.accuracy2(predicted_labels=predicted_labels, correct_labels=correct_labels)


class MultiClassAnalyzer(Analyzer):
    def __init__(self, csv_path):
        super().__init__(csv_path)

    def accuracy(self, predicted_labels, correct_labels):
        return self.accuracy2(predicted_labels=predicted_labels, correct_labels=correct_labels)




