from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
from barbar import Bar
import time
import os
import copy
import sys
import Pytorch_custom_dataloader as custom_data
import logging
import yaml
import logging.config

import sys

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def setup_logging(default_path='log_config.yaml', logName='info.log', default_level=logging.DEBUG):
    path = default_path
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            config["handlers"]["file"]['filename'] = logName
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

yaml_path = 'log_config.yaml'
setup_logging(yaml_path, logName="train.log")

logger = logging.getLogger("main.core")

if __name__ == '__main__':
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)
    data_dir = r"K:\OneDrive_USC\OneDrive - University of South Carolina\Research\noise_map"

    # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception, efficientnet_b6]
    model_name = "vgg"

    # Number of classes in the dataset
    num_classes = 7

    # Batch size for training (change depending on how much memory you have)
    if len(sys.argv) > 1:
        batch_size = 2 ** int(sys.argv[1])
    else:
        batch_size = 2 ** 3

    # print("Batch size:", batch_size)

    batch_size = 8

    print("Batch size:", batch_size)

    # Number of epochs to train for
    num_epochs = 15

    # Flag for feature extracting. When False, we finetune the whole model,
    #   when True we only update the reshaped layer params
    feature_extract = False


    def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
        since = time.time()

        val_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0

        try:
            for epoch in range(num_epochs):
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()  # Set model to training mode
                    else:
                        model.eval()  # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data.
                    all_paths = []
                    all_preds = []
                    all_labels = []
                    for idx, (inputs, labels, paths) in enumerate(Bar(dataloaders[phase])):
                        # print(paths)
                        all_paths += paths
                        all_labels += list(labels.cpu().detach().numpy())

                        try:
                            inputs = inputs.to(device, dtype=torch.float)
                            labels = labels.to(device)

                            # zero the parameter gradients
                            optimizer.zero_grad()

                            # forward
                            # track history if only in train
                            with torch.set_grad_enabled(phase == 'train'):
                                # Get model outputs and calculate loss
                                # Special case for inception because in training it has an auxiliary output. In train
                                #   mode we calculate the loss by summing the final output and the auxiliary output
                                #   but in testing we only consider the final output.
                                if is_inception and phase == 'train':
                                    # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                                    outputs, aux_outputs = model(inputs)
                                    loss1 = criterion(outputs, labels)
                                    loss2 = criterion(aux_outputs, labels)
                                    loss = loss1 + 0.4 * loss2
                                else:
                                    outputs = model(inputs)
                                    loss = criterion(outputs, labels)

                                _, preds = torch.max(outputs, 1)
                                all_preds += list(preds.cpu().detach().numpy())

                                previous_cnt = len(all_preds) - len(outputs)
                                # for idx, output in enumerate(outputs):
                                #     current_idx = idx + previous_cnt
                                #     output = output.cpu().detach().numpy()
                                #     basename = os.path.basename(all_paths[current_idx])
                                    # logger.info("image: %s", basename)
                                    # logger.info("output: %s", output)
                                    # logger.info("pred: %s", all_preds[current_idx])
                                    # logger.info("label: %s", all_labels[current_idx])

                                # backward + optimize only if in training phase
                                if phase == 'train':
                                    loss.backward()
                                    optimizer.step()

                            # statistics
                            running_loss += loss.item() * inputs.size(0)
                            running_corrects += torch.sum(preds == labels.data)

                            if idx % 1000 == 0:
                                step_loss =  loss.item()
                                step_acc = 1.0 * torch.sum(preds == labels.data) / len(inputs)

                                print('\n{} Step-Loss: {:.4f} Step-Acc: {:.4f}'.format(os.path.basename(phase), step_loss,
                                                                           step_acc))

                                if idx % 2000 == 0:
                                    torch.save(model.state_dict(), PATH)
                                    torch.save(model, PATH_ENTIRE_MODEL)

                        except Exception as e:
                            print("Error in minibatch: ", idx, paths,  e)
                            exception_type, exception_object, exception_traceback = sys.exc_info()
                            filename = exception_traceback.tb_frame.f_code.co_filename
                            line_number = exception_traceback.tb_lineno

                            print("Exception type: ", exception_type)
                            print("File name: ", filename)
                            print("Line number: ", line_number)
                            continue

                    epoch_loss = running_loss / len(dataloaders[phase].dataset)
                    epoch_acc = 1.0 * running_corrects / len(dataloaders[phase].dataset)

                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(os.path.basename(phase), epoch_loss, epoch_acc))

                    # deep copy the model
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                        torch.save(model.state_dict(), PATH)

                        torch.save(model, PATH_ENTIRE_MODEL)

                        print("Better model found. saving")
                    if phase == 'val':
                        print(f"Current best: {best_acc}")
                        val_acc_history.append(epoch_acc)

                    # f = open(phase + '_predicts.csv', 'w')
                    # f.writelines("image,predict,label\n")
                    # for idx, path in enumerate(all_paths):
                    #     f.writelines("{},{},{}\n".format(path,  all_preds[idx], all_labels[idx]))
                    # f.close()

                print()
        except KeyboardInterrupt:
            torch.save(model.state_dict(), PATH)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        return model, val_acc_history


    def set_parameter_requires_grad(model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False


    PATH = r"{}.pth".format(model_name)
    PATH_ENTIRE_MODEL = r"{}_all.pth".format(model_name)

    def vgg_regression():

        return None

    def initialize_model(model_name, num_classes, feature_extract, use_pretrained=False):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0

        if model_name == "efficientnet_b6":

            model_ft = models.efficientnet_b6(pretrained=use_pretrained)
            # print("EfficientNet:", model_ft)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs =model_ft.classifier[1].in_features
            model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

            print("efficientnet_b6 model:", model_ft)

        elif model_name == "resnet":
            """ Resnet18
            """
            model_ft = models.resnet101(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

            print("ResNet model:", model_ft)

        elif model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg16_bn(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            # print("VGG model,  model_ft.classifier[6].in_features:",  model_ft.classifier[0].in_features)
            # print("VGG model, features:",  model_ft.features)
            num_ftrs = model_ft.classifier[6].in_features
            # model_ft.classifier[0] = nn.Linear(512 * 16 * 64, 1024)
            # model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

            print("VGG model:", model_ft)

        elif model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = num_classes
            input_size = 224

        elif model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet121(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 299

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size


    # # Initialize the model for this run
    model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=False)


    # print("Model_ft: ", model_ft)

    #
    if os.path.exists(PATH):
        model_ft.load_state_dict(torch.load(PATH))

    input_size = (int(256*1), int(1024*1))
    # input_size = (int(256*2), int(1024*2))

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5)
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(input_size),
            # transforms.CenterCrop(input_size),
            transforms.ToTensor()
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    image_datasets = {x: custom_data.ImageListDataset(os.path.join(data_dir, x + '.csv'), r'K:\Research\Noise_map\panoramas4_jpg_half',
                                                      transform=data_transforms[x]) for x in ['train', 'val']}  # huan

    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=10) for x in
        ['train', 'val']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'

    # Send the model to GPU
    device = 'cuda:0'
    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)
                pass

    # Observe that all parameters are being optimized0
    optimizer_ft = optim.SGD(params_to_update, lr=0.00001, momentum=0.9)
    # optimizer_ft = optim.Adam(params_to_update, lr=0.0001)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs,
                                 is_inception=(model_name == 'inception'))

