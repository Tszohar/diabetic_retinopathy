import torch
from torch.utils.data import DataLoader
import os
import parameters
from convert_model2csv import model2csv
from data import BDDataset
from network import BDNetwork, Outputs
from resnet import ResNet

if __name__ == '__main__':

    batch_size = 32
    classifier_type = Outputs.MULTI_CLASS

    model_path = "/media/guy/Files 3/Tsofit/blindness detection/results/20191111 (16:28:16.076528)_MULTI_CLASS_64_HealthyAndSick/model_epoch_21.pth"
    output_path = os.path.join(os.path.dirname(model_path), 'analysis_dir_train', 'analysis.csv')
    statistics_path = os.path.join(os.path.dirname(model_path), 'analysis_dir', 'statistics.csv')
    mislabeled_path = os.path.join(os.path.dirname(model_path), 'analysis_dir', 'mislabeled_images.csv')
    model = torch.load(model_path)
    net = ResNet(num_outputs=5) #BDNetwork(classifier_type)
    net.load_state_dict(model, strict=False)
    net.to(parameters.device)

    dataset = BDDataset(csv_file=parameters.train_csv, data_dir=parameters.data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # Converting model output from dataset to csv if it doesn't exists
    if os.path.isfile(output_path):
        pass
    else:
        model2csv(parameters.validation_csv, parameters.data_dir, model_path, output_path, classifier_type)

    # Analyzing csv
    analyzer = parameters.analyzer_dict[classifier_type](output_path)
    predicted_labels = analyzer.dataframe['predicted_label'][:, None]
    converted_label = analyzer.dataframe['converted_label'][:, None]
    num_of_classes = parameters.num_classes_dict[classifier_type.name]

    analyzer.statistics(num_of_classes=num_of_classes, predicted_labels=predicted_labels,
                        correct_labels=converted_label, output_path=statistics_path)
    mislabeled_idxs = analyzer.mislabeled_images(predicted_labels=predicted_labels,
                                                 correct_labels=converted_label, output_path=mislabeled_path)
    analyzer.count_mislabeled_by_class(num_of_classes=num_of_classes, output_path=mislabeled_path)

    images_by_class = analyzer.count_images_by_class(num_of_classes=num_of_classes)


    # image handling

    # copy_images = CopyImages()
    # dataset2csv = ConvertDataset2Csv()
    # for i_batch, sample_batched in enumerate(dataloader):
        # outputs = net(sample_batched['image'].to(parameters.device))
        # copy_images.by_predicate(dataset=sample_batched, predicate=outputs)
        # dataset2csv.convert(dataset=sample_batched, predicate=outputs)




