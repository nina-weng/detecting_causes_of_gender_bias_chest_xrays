import pytorch_lightning as pl
from torchvision import models
import torch.nn as nn
import torch
import torchvision
import torch.nn.functional as F
from torchmetrics import Accuracy,AUROC
from torchmetrics.classification import MultilabelAUROC


class ResNet(pl.LightningModule):
    def __init__(self, num_classes,lr,pretrained,model_scale,loss_func_type='BCE'):
        super().__init__()
        self.model_name = 'resnet'
        self.num_classes = num_classes
        self.pretrained=pretrained
        self.model_scale = model_scale
        self.loss_func_type= loss_func_type
        if self.model_scale == '18':
            self.model = models.resnet18(pretrained=self.pretrained)
        elif self.model_scale == '34':
            self.model = models.resnet34(pretrained=self.pretrained)
        elif self.model_scale == '50':
            self.model = models.resnet50(pretrained=self.pretrained)
        else:
            raise Exception('not implemented model scale: '+model_scale)

        # freeze_model(self.model)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.num_classes)

        self.lr=lr
        if self.loss_func_type == 'BCE':
            # self.loss_func = F.binary_cross_entropy
            self.loss_func = nn.BCELoss()
        elif self.loss_func_type == 'WeightedBCE':
            pos_weight = torch.tensor([100.0])

            # Define the loss function with weighted binary cross-entropy
            self.loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            raise Exception('Not implemented loss function type : {}'.format(self.loss_func_type))

        if self.num_classes == 1:
            self.accu_func = Accuracy(task="binary", num_labels=num_classes)
            self.auroc_func = AUROC(task='binary',num_labels=num_classes, average='macro', thresholds=None)
        elif self.num_classes >1:
            self.accu_func= Accuracy(task="multilabel", num_labels=num_classes)
            self.auroc_func = MultilabelAUROC(num_labels=num_classes,average='macro', thresholds=None)

    def remove_head(self):
        num_features = self.model.fc.in_features
        id_layer = nn.Identity(num_features)
        self.model.fc = id_layer

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=self.lr)
        return optimizer

    def unpack_batch(self, batch):
        return batch['image'], batch['label']

    def process_batch(self, batch):
        img, lab = self.unpack_batch(batch)
        out = self.forward(img)
        prob = torch.sigmoid(out)
        loss = self.loss_func(prob, lab)

        multi_accu = self.accu_func(prob, lab)
        multi_auroc = self.auroc_func(prob,lab.long())
        # print(prob.shape,lab.shape,lab.long().shape)
        # print('multi_auroc:{:.4f}'.format(multi_auroc))
        return loss,multi_accu,multi_auroc

    def training_step(self, batch, batch_idx):
        loss,multi_accu,multi_auroc = self.process_batch(batch)
        self.log('train_loss', loss)
        self.log('train_accu', multi_accu)
        self.log('train_auroc', multi_auroc)
        # grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
        # self.logger.experiment.add_image('images', grid, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, multi_accu, multi_auroc = self.process_batch(batch)
        self.log('val_loss', loss)
        self.log('val_accu', multi_accu)
        self.log('val_auroc', multi_auroc)

    def test_step(self, batch, batch_idx):
        loss,multi_accu,multi_auroc = self.process_batch(batch)
        self.log('test_loss', loss)
        self.log('test_accu', multi_accu)
        self.log('test_auroc', multi_auroc)


class DenseNet(pl.LightningModule):
    def __init__(self, num_classes,lr,pretrained,model_scale='121'):
        super().__init__()
        self.model_name = 'densenet'
        self.lr = lr
        self.pretrained = pretrained
        self.num_classes = num_classes
        self.model_scale= model_scale
        if self.model_scale == '121':
            self.model = models.densenet121(pretrained=self.pretrained)
        else:
            raise Exception('not implemented model scale: '+model_scale)

        # freeze_model(self.model)
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, self.num_classes)

        self.accu_func = Accuracy(task="multilabel", num_labels=num_classes)
        self.auroc_func = MultilabelAUROC(num_labels=num_classes, average='macro', thresholds=None)

    def remove_head(self):
        num_features = self.model.classifier.in_features
        id_layer = nn.Identity(num_features)
        self.model.classifier = id_layer

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=self.lr)
        return optimizer

    def unpack_batch(self, batch):
        return batch['image'], batch['label']

    def process_batch(self, batch):
        img, lab = self.unpack_batch(batch)
        out = self.forward(img)
        prob = torch.sigmoid(out)
        loss = F.binary_cross_entropy(prob, lab)

        multi_accu = self.accu_func(prob, lab)
        multi_auroc = self.auroc_func(prob, lab.long())
        return loss, multi_accu, multi_auroc

    def training_step(self, batch, batch_idx):
        loss, multi_accu, multi_auroc = self.process_batch(batch)
        self.log('train_loss', loss)
        self.log('train_accu', multi_accu)
        self.log('train_auroc', multi_auroc)
        # grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
        # self.logger.experiment.add_image('images', grid, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, multi_accu, multi_auroc = self.process_batch(batch)
        self.log('val_loss', loss)
        self.log('val_accu', multi_accu)
        self.log('val_auroc', multi_auroc)

    def test_step(self, batch, batch_idx):
        loss, multi_accu, multi_auroc = self.process_batch(batch)
        self.log('test_loss', loss)
        self.log('test_accu', multi_accu)
        self.log('test_auroc', multi_auroc)

    def test_step_end(self, output_results):
        prob, lab = output_results
        loss = F.binary_cross_entropy(prob, lab)

        multi_accu = self.accu_func(prob, lab)
        multi_auroc = self.auroc_func(prob, lab.long())
        self.log('test_loss', loss,epoch_end=True)
        self.log('test_accu', multi_accu,epoch_end=True)
        self.log('test_auroc', multi_auroc,epoch_end=True)
