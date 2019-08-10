from sklearn.metrics import confusion_matrix
import pandas as pd


def calculate_kappa(correct_label, predicted_label):
    ##############################################################################
    # Kappa = (observed accuracy - expected accuracy) / (1 - expected accuracy)
    ##############################################################################
    conf_matrix = confusion_matrix(correct_label, predicted_label)
    num_categories = int(conf_matrix.size / 2)
    total_values = conf_matrix.sum()
    observed_accuracy = sum(conf_matrix.diagonal())/total_values
    expected_accuracy = 0.0
    for i in range(num_categories):
        ground_truth = sum(conf_matrix[:,i])
        model_prediction = sum(conf_matrix[i, :])
        expected_accuracy += (ground_truth * model_prediction) / total_values

    expected_accuracy /= total_values
    kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)
    return kappa


if __name__ == '__main__':
    output_path = '/home/guy/tsofit/blindness detection/dataset29-7.csv'
    dataframe = pd.read_csv(output_path)
    kappa = calculate_kappa(dataframe['label'], dataframe['predicted_label'])
    print(kappa)
