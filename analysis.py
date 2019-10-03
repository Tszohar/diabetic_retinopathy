import torch
from torch.utils.data import DataLoader
import os
import parameters
from convert_model2csv import model2csv
from data import BDDataset

from network import BDNetwork, Outputs
from post_process import Analyzer, RegressorAnalyzer, MultiBinaryAnalyzer

if __name__ == '__main__':

    batch_size = 32
    classifier_type = Outputs.BINARY

    model_path = "/media/guy/Files 3/Tsofit/blindness detection/results/20190919 (11:56:16.840499)_BINARY_32/model_epoch_99.pth"
    output_path = os.path.join(os.path.dirname(model_path), 'analysis_dir', 'analysis.csv')
    statistics_path = os.path.join(os.path.dirname(model_path), 'analysis_dir', 'statistics.csv')
    mislabeled_path = os.path.join(os.path.dirname(model_path), 'analysis_dir', 'mislabeled_images.csv')
    model = torch.load(model_path)
    net = BDNetwork(classifier_type)
    net.load_state_dict(model)
    net.to(parameters.device)

    dataset = BDDataset(csv_file=parameters.validation_csv, data_dir=parameters.data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # Converting model output from dataset to csv if it doesn't exists
    if os.path.isfile(output_path):
        pass
    else:
        model2csv(parameters.validation_csv, parameters.data_dir, model_path, output_path, classifier_type)

    # Analyzing csv
    analyzer = MultiBinaryAnalyzer(output_path)
    predicted_labels = analyzer.dataframe['predicted_label'][:, None]
    converted_label = analyzer.dataframe['converted_label'][:, None]
    num_of_classes = parameters.num_classes_dict[classifier_type.name]

    accuracy = analyzer.accuracy(predicted_labels=predicted_labels,
                                 correct_labels=converted_label)
    mislabeled_idxs = analyzer.mislabeled_images(predicted_labels=predicted_labels,
                                                 correct_labels=converted_label, output_path=mislabeled_path)
    analyzer.count_mislabeled_by_class(num_of_classes=num_of_classes, output_path=mislabeled_path)
    analyzer.statistics(num_of_classes=num_of_classes, predicted_labels=predicted_labels,
                        correct_labels=converted_label, output_path=statistics_path)

    images_by_class = analyzer.count_images_by_class(num_of_classes=num_of_classes)

    # image handling


    # copy_images = CopyImages()
    # dataset2csv = ConvertDataset2Csv()
    # for i_batch, sample_batched in enumerate(dataloader):
        # outputs = net(sample_batched['image'].to(parameters.device))
        # copy_images.by_predicate(dataset=sample_batched, predicate=outputs)
        # dataset2csv.convert(dataset=sample_batched, predicate=outputs)


    # splited_model_path = os.path.split(model_path, '/')
    # output_csv = os.path.join(parameters.analysis_dir, splited_model_path[6], splited_model_path[7])
    # #Converting the dataset into csv file
    # model2csv(parameters.validation_csv, parameters.data_dir, model_path, output_csv)
    #
    # dataframe = pd.read_csv(output_csv)
    # copy_images_to_subfolder(dataframe, analysis_dir=parameters.analysis_dir)
    #
    # # Save mislabeled CSV
    # mis_df = dataframe[dataframe['predicted_label'] != dataframe['label']]
    # mis_df.to_csv(os.path.join(parameters.analysis_dir, "mislabeled.csv"))
    #
    # recall_by_class(dataframe)
    # accuracy_by_class(dataframe)
    # conf_matrix = confusion_matrix(dataframe['label'], dataframe['predicted_label'])
    # print('confusion matrix:')
    # print(conf_matrix)
    # kappa_score = cohen_kappa_score(dataframe['label'], dataframe['predicted_label'])
    # print('kappa score: {}'.format(kappa_score))
    #
    # loss = loss_by_sample(dataframe)

