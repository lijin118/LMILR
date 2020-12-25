#vaemodel
import copy
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.utils import data
from data_loader import DATA_LOADER as dataloader
import final_classifier as  classifier
import models
import final_classifier2 as  classifier2
from scipy.stats import entropy
from gan_losses import get_positive_expectation, get_negative_expectation
import sys
from data_loader import map_label
import numpy as np
import math



class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim,nclass)
        self.logic = nn.LogSoftmax(dim=1)
        self.lossfunction =  nn.NLLLoss()

    def forward(self, x):
        o = self.logic(self.fc(x))
        return o

class Model(nn.Module):

    def __init__(self,hyperparameters,args):
        super(Model,self).__init__()
        self.args = args
        self.device = hyperparameters['device']
        self.auxiliary_data_source = hyperparameters['auxiliary_data_source']
        self.all_data_sources  = ['resnet_features',self.auxiliary_data_source]
        self.DATASET = hyperparameters['dataset']
        self.num_shots = hyperparameters['num_shots']
        self.latent_size = hyperparameters['latent_size']
        self.batch_size = hyperparameters['batch_size']
        self.hidden_size_rule = hyperparameters['hidden_size_rule']
        self.warmup = hyperparameters['model_specifics']['warmup']
        self.generalized = hyperparameters['generalized']
        self.classifier_batch_size = self.args.cls_batch_size
        self.img_seen_samples   = hyperparameters['samples_per_class'][self.DATASET][0]
        self.att_seen_samples   = hyperparameters['samples_per_class'][self.DATASET][1]
        self.att_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][2]
        self.img_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][3]
        self.reco_loss_function = hyperparameters['loss']
        self.nepoch = hyperparameters['epochs']
        self.lr_cls = hyperparameters['lr_cls']
        self.cross_reconstruction = hyperparameters['model_specifics']['cross_reconstruction']
        self.cls_train_epochs = hyperparameters['cls_train_steps']
        self.dataset = dataloader( self.DATASET, copy.deepcopy(self.auxiliary_data_source) , device= self.device )

        self.softmax_layer = nn.Softmax(dim=1).cuda()

        if self.DATASET=='CUB':
            self.num_classes=200
            self.num_novel_classes = 50
        elif self.DATASET=='SUN':
            self.num_classes=717
            self.num_novel_classes = 72
        elif self.DATASET=='AWA1' or self.DATASET=='AWA2':
            self.num_classes=50
            self.num_novel_classes = 10
        elif self.DATASET=='APY':
            self.num_classes=32
            self.num_novel_classes = 12
        elif self.DATASET=='FLO':
            self.num_classes=102
            self.num_novel_classes = 20

        feature_dimensions = [2048, self.dataset.aux_data.size(1)]

        self.encoder = {}

        for datatype, dim in zip(self.all_data_sources,feature_dimensions):

            self.encoder[datatype] = models.encoder_template(dim,self.latent_size,self.hidden_size_rule[datatype],self.device)

            print(str(datatype) + ' ' + str(dim))

        self.decoder = {}
        for datatype, dim in zip(self.all_data_sources,feature_dimensions):
            self.decoder[datatype] = models.decoder_template(self.latent_size,dim,self.hidden_size_rule[datatype],self.device)

        # An optimizer for all encoders and decoders is defined here
        parameters_to_optimize = list(self.parameters())
        for datatype in self.all_data_sources:
            parameters_to_optimize +=  list(self.encoder[datatype].parameters())
            parameters_to_optimize +=  list(self.decoder[datatype].parameters())

        self.optimizer  = optim.Adam( parameters_to_optimize ,lr=hyperparameters['lr_gen_model'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

        if self.reco_loss_function=='l2':
            self.reconstruction_criterion = nn.MSELoss(size_average=False)

        elif self.reco_loss_function=='l1':
            self.reconstruction_criterion = nn.L1Loss(size_average=False)

    def reparameterize(self, mu, logvar):
        if self.reparameterize_with_noise:
            sigma = torch.exp(logvar)
            eps = torch.cuda.FloatTensor(logvar.size()[0],1).normal_(0,1)
            eps  = eps.expand(sigma.size())
            return mu + sigma*eps
        else:
            return mu

    def forward(self):
        pass

    def map_label(self,label, classes):
        mapped_label = torch.LongTensor(label.size()).to(self.device)
        for i in range(classes.size(0)):
            mapped_label[label==classes[i]] = i

        return mapped_label

    def fenchel_dual_loss(self,l, m, measure=None):
        '''Computes the f-divergence distance between positive and negative joint distributions.

        Note that vectors should be sent as 1x1.

        Divergences supported are Jensen-Shannon `JSD`, `GAN` (equivalent to JSD),
        Squared Hellinger `H2`, Chi-squeared `X2`, `KL`, and reverse KL `RKL`.

        Args:
            l: Local feature map.
            m: Multiple globals feature map.
            measure: f-divergence measure.

        Returns:
            torch.Tensor: Loss.

        '''
        l = l.unsqueeze(1)
        m = m.unsqueeze(1)


        N, units, n_locals = l.size()
        n_multis = m.size(2)

        # First we make the input tensors the right shape.
        l = l.view(N, units, n_locals)
        l = l.permute(0, 2, 1)
        l = l.reshape(-1, units)

        m = m.view(N, units, n_multis)
        m = m.permute(0, 2, 1)
        m = m.reshape(-1, units)

        # Outer product, we want a N x N x n_local x n_multi tensor.
        u = torch.mm(m, l.t())
        u = u.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1)

        # Since we have a big tensor with both positive and negative samples, we need to mask.
        mask = torch.eye(N).to(l.device)
        n_mask = 1 - mask

        # Compute the positive and negative score. Average the spatial locations.
        E_pos = get_positive_expectation(u, measure, average=False).mean(2).mean(2)
        E_neg = get_negative_expectation(u, measure, average=False).mean(2).mean(2)

        # Mask positive and negative terms for positive and negative parts of loss
        E_pos = (E_pos * mask).sum() / mask.sum()
        E_neg = (E_neg * n_mask).sum() / n_mask.sum()
        loss = E_neg - E_pos

        return loss

    def sample_unseen_att(self):
        unseen_att_all = self.dataset.aux_data[self.dataset.novelclasses]
        unseen_label_all = self.dataset.novelclasses
        if unseen_att_all.size()[0] >= self.batch_size:
            att_idx = torch.randperm(unseen_att_all.size()[0])[:self.batch_size]
            return unseen_att_all[att_idx,:], unseen_label_all[att_idx]
        elif unseen_att_all.size()[0] < self.batch_size:
            multi_unseen_att_all = unseen_att_all
            multi_unseen_label_all = unseen_label_all
            while(multi_unseen_att_all.size()[0] < self.batch_size):
                multi_unseen_att_all = torch.cat((multi_unseen_att_all,unseen_att_all),0)
                multi_unseen_label_all = torch.cat((multi_unseen_label_all,unseen_label_all),0)
            return multi_unseen_att_all[:self.batch_size,:], multi_unseen_label_all[:self.batch_size]


    def EntropicConfusion(self, z):
        softmax_out = nn.Softmax(dim=1)(z)
        batch_size = z.size(0)
        loss = torch.mul(softmax_out, torch.log(softmax_out)).sum() * (1.0 / batch_size)
        return loss

    def mytrainstep(self, img, att):
        mu_img, logvar_img = self.encoder['resnet_features'](img)
        z_from_img = self.reparameterize(mu_img, logvar_img)

        mu_att, logvar_att = self.encoder[self.auxiliary_data_source](att)
        z_from_att = self.reparameterize(mu_att, logvar_att)

        ##############################################
        # Reconstruct inputs
        ##############################################

        img_from_img = self.decoder['resnet_features'](z_from_img)
        att_from_att = self.decoder[self.auxiliary_data_source](z_from_att)

        reconstruction_loss = self.reconstruction_criterion(img_from_img, img) \
                              + self.reconstruction_criterion(att_from_att, att)

        ##############################################
        # Cross Reconstruction Loss
        ##############################################
        img_from_att = self.decoder['resnet_features'](z_from_att)
        att_from_img = self.decoder[self.auxiliary_data_source](z_from_img)

        cross_reconstruction_loss = self.reconstruction_criterion(img_from_att, img) \
                                    + self.reconstruction_criterion(att_from_img, att)

        ##############################################
        # KL-Divergence
        ##############################################

        KLD = (0.5 * torch.sum(1 + logvar_att - mu_att.pow(2) - logvar_att.exp())) \
              + (0.5 * torch.sum(1 + logvar_img - mu_img.pow(2) - logvar_img.exp()))

        ##############################################
        # Distribution Alignment
        ##############################################
        distance = torch.sqrt(torch.sum((mu_img - mu_att) ** 2, dim=1) + \
                              torch.sum((torch.sqrt(logvar_img.exp()) - torch.sqrt(logvar_att.exp())) ** 2, dim=1))

        distance = distance.sum()

        ##############################################
        # scale the loss terms according to the warmup
        # schedule
        ##############################################

        f1 = 1.0*(self.current_epoch - self.warmup['cross_reconstruction']['start_epoch'] )/(1.0*( self.warmup['cross_reconstruction']['end_epoch']- self.warmup['cross_reconstruction']['start_epoch']))
        f1 = f1*(1.0*self.warmup['cross_reconstruction']['factor'])
        cross_reconstruction_factor = torch.cuda.FloatTensor([min(max(f1,0),self.warmup['cross_reconstruction']['factor'])])

        f2 = 1.0 * (self.current_epoch - self.warmup['beta']['start_epoch']) / ( 1.0 * (self.warmup['beta']['end_epoch'] - self.warmup['beta']['start_epoch']))
        f2 = f2 * (1.0 * self.warmup['beta']['factor'])
        beta = torch.cuda.FloatTensor([min(max(f2, 0), self.warmup['beta']['factor'])])

        f3 = 1.0*(self.current_epoch - self.warmup['distance']['start_epoch'] )/(1.0*( self.warmup['distance']['end_epoch']- self.warmup['distance']['start_epoch']))
        f3 = f3*(1.0*self.warmup['distance']['factor'])
        distance_factor = torch.cuda.FloatTensor([min(max(f3,0),self.warmup['distance']['factor'])])

        ##############################################
        # Put the loss together and call the optimizer
        ##############################################

        self.optimizer.zero_grad()

        # VAE loss
        loss = reconstruction_loss - beta * KLD

        # CA loss
        if cross_reconstruction_loss>0:
            loss += cross_reconstruction_factor*cross_reconstruction_loss

        # DA loss
        distance_factor = self.args.da_weight * distance_factor
        if distance_factor >0:
            loss += distance_factor*distance

        ###############Mutual Information################
        max_entropy_loss = torch.FloatTensor([0])
        max_loss = torch.FloatTensor([0])
        min_loss = torch.FloatTensor([0])

        if self.args.entr_weight > 0:
            joint_z = torch.cat((z_from_att, z_from_img),0)
            max_entropy_loss = self.args.entr_weight * self.EntropicConfusion(joint_z)
            loss += max_entropy_loss

        if self.args.max_weight > 0:
            mi_loss_recon = self.fenchel_dual_loss(z_from_att, z_from_img,'JSD')
            max_loss = self.args.max_weight * mi_loss_recon
            loss -= max_loss

        if self.args.min_weight > 0:
            unseen_att, unseen_label = self.sample_unseen_att()
            unseen_mu_att, unseen_logvar_att = self.encoder[self.auxiliary_data_source](unseen_att)
            unseen_z_from_att = self.reparameterize(unseen_mu_att, unseen_logvar_att)
            mi_loss_cross = self.fenchel_dual_loss(unseen_z_from_att, z_from_att,'JSD')
            min_loss = self.args.min_weight * mi_loss_cross
            loss += min_loss


        loss.backward()

        self.optimizer.step()

        return loss.item(), max_loss.item(), min_loss.item(), max_entropy_loss.item()

    def my_train_vae(self):
        losses = []
        self.dataset.novelclasses =self.dataset.novelclasses.long().cuda()
        self.dataset.seenclasses =self.dataset.seenclasses.long().cuda()
        self.train()
        self.reparameterize_with_noise = True

        print('train for reconstruction')
        test_loss = 0
        for epoch in range(0, self.nepoch ):
            self.current_epoch = epoch
            i=-1
            for iters in range(0, self.dataset.ntrain, self.batch_size):
                i+=1
                label, data_from_modalities = self.dataset.next_batch(self.batch_size)
                label= label.long().to(self.device)
                for j in range(len(data_from_modalities)):
                    data_from_modalities[j] = data_from_modalities[j].to(self.device)
                    data_from_modalities[j].requires_grad = False
                loss,max_loss,min_loss,max_entropy_loss = self.mytrainstep(data_from_modalities[0], data_from_modalities[1] )
                test_loss = loss
                if i%50==0:
                    print('epoch ' + str(epoch) + ' | iter ' + str(i) + '\t'+
                    ' | loss ' +  str(loss)[:5] + '  |max ' + str(max_loss)+ '  |min ' + str(min_loss) + '  |entr ' + str(max_entropy_loss)   )

                if i%50==0 and i>0:
                    losses.append(loss)
            sys.stdout.flush()

            if math.isnan(test_loss):
                return 'nan'

        # turn into evaluation mode:
        for key, value in self.encoder.items():
            self.encoder[key].eval()
        for key, value in self.decoder.items():
            self.decoder[key].eval()

        return losses

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        per_class_accuracies = torch.zeros(target_classes.size()[0]).float().to(self.device).detach()
        predicted_label = predicted_label.to(self.device)
        for i in range(target_classes.size()[0]):
            is_class = test_label==target_classes[i]
            per_class_accuracies[i] = torch.div((predicted_label[is_class]==test_label[is_class]).sum().float(),is_class.sum().float())
        return per_class_accuracies.mean()


    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        per_class_accuracies = torch.zeros(nclass).float().to(self.device).detach()
        target_classes = torch.arange(0, nclass, out=torch.LongTensor()).to(self.device) #changed from 200 to nclass on 24.06.
        # predicted_label = predicted_label.to(self.device)
        # test_label = test_label.to(self.device)
        for i in range(nclass):
            is_class = test_label==target_classes[i]
            per_class_accuracies[i] = torch.div((predicted_label[is_class]==test_label[is_class]).sum().float(),is_class.sum().float())
        return per_class_accuracies.mean()

    

    def fuse_pred(self,output1, output2, all_label):
        all_output = output1 + 0.7 * output2
        all_pred = torch.argmax(all_output.data, 1)
        acc_seen = self.compute_per_class_acc_gzsl(all_label, all_pred, self.dataset.seenclasses)
        acc_unseen = self.compute_per_class_acc_gzsl(all_label, all_pred, self.dataset.novelclasses)
        if (acc_seen == 0) or (acc_unseen == 0):
            acc_H = 0
        else:
            acc_H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
        return acc_seen*100, acc_unseen*100, acc_H*100

    def mytrain_classifier(self, show_plots=False):
        cls_seenclasses = self.dataset.seenclasses
        cls_novelclasses = self.dataset.novelclasses

        train_seen_feat = self.dataset.data['train_seen']['resnet_features']
        train_seen_label = self.dataset.data['train_seen']['labels']

        novelclass_aux_data = self.dataset.novelclass_aux_data  # access as novelclass_aux_data['resnet_features'], novelclass_aux_data['attributes']
        seenclass_aux_data = self.dataset.seenclass_aux_data

        novel_corresponding_labels = self.dataset.novelclasses.long().to(self.device)
        seen_corresponding_labels = self.dataset.seenclasses.long().to(self.device)


        # The resnet_features for testing the classifier are loaded here
        novel_test_feat = self.dataset.data['test_unseen'][
            'resnet_features']  # self.dataset.test_novel_feature.to(self.device)
        seen_test_feat = self.dataset.data['test_seen'][
            'resnet_features']  # self.dataset.test_seen_feature.to(self.device)
        test_seen_label = self.dataset.data['test_seen']['labels']  # self.dataset.test_seen_label.to(self.device)
        test_novel_label = self.dataset.data['test_unseen']['labels']  # self.dataset.test_novel_label.to(self.device)

        train_unseen_feat = self.dataset.data['train_unseen']['resnet_features']
        train_unseen_label = self.dataset.data['train_unseen']['labels']


        # in ZSL mode:
        if self.generalized == False:
            # there are only 50 classes in ZSL (for CUB)
            # novel_corresponding_labels =list of all novel classes (as tensor)
            # test_novel_label = mapped to 0-49 in classifier function
            # those are used as targets, they have to be mapped to 0-49 right here:

            novel_corresponding_labels = self.map_label(novel_corresponding_labels, novel_corresponding_labels)

            if self.num_shots > 0:
                # not generalized and at least 1 shot means normal FSL setting (use only unseen classes)
                train_unseen_label = self.map_label(train_unseen_label, cls_novelclasses)

            # for FSL, we train_seen contains the unseen class examples
            # for ZSL, train seen label is not used
            # if self.num_shots>0:
            #    train_seen_label = self.map_label(train_seen_label,cls_novelclasses)

            test_novel_label = self.map_label(test_novel_label, cls_novelclasses)

            # map cls novelclasses last
            cls_novelclasses = self.map_label(cls_novelclasses, cls_novelclasses)


        if self.generalized:
            print('mode: gzsl')
            clf = LINEAR_LOGSOFTMAX(self.latent_size, self.num_classes)
            clf3 = LINEAR_LOGSOFTMAX(self.dataset.aux_data.size(1), self.num_classes)
        else:
            print('mode: zsl')
            clf = LINEAR_LOGSOFTMAX(self.latent_size, self.num_novel_classes)
            clf3 = LINEAR_LOGSOFTMAX(self.dataset.aux_data.size(1), self.num_novel_classes)

        clf.apply(models.weights_init)
        clf3.apply(models.weights_init)


        with torch.no_grad():

            ####################################
            # preparing the test set
            # convert raw test data into z vectors
            ####################################

            self.reparameterize_with_noise = False

            mu1, var1 = self.encoder['resnet_features'](novel_test_feat)
            test_novel_X = self.reparameterize(mu1, var1).to(self.device).data
            test_novel_Y = test_novel_label.to(self.device)

            mu2, var2 = self.encoder['resnet_features'](seen_test_feat)
            test_seen_X = self.reparameterize(mu2, var2).to(self.device).data
            test_seen_Y = test_seen_label.to(self.device)

            all_X = torch.cat((test_novel_X, test_seen_X), 0)
            all_att = self.decoder[self.auxiliary_data_source](all_X)
            test_unseen_att = self.decoder[self.auxiliary_data_source](test_novel_X)
            ####################################
            # preparing the train set:
            # chose n random image features per
            # class. If n exceeds the number of
            # image features per class, duplicate
            # some. Next, convert them to
            # latent z features.
            ####################################

            self.reparameterize_with_noise = True

            def sample_train_data_on_sample_per_class_basis(features, label, sample_per_class):
                sample_per_class = int(sample_per_class)

                if sample_per_class != 0 and len(label) != 0:

                    classes = label.unique()

                    for i, s in enumerate(classes):

                        features_of_that_class = features[label == s, :]  # order of features and labels must coincide
                        # if number of selected features is smaller than the number of features we want per class:
                        multiplier = torch.ceil(torch.cuda.FloatTensor(
                            [max(1, sample_per_class / features_of_that_class.size(0))])).long().item()

                        features_of_that_class = features_of_that_class.repeat(multiplier, 1)

                        if i == 0:
                            features_to_return = features_of_that_class[:sample_per_class, :]
                            labels_to_return = s.repeat(sample_per_class)
                        else:
                            features_to_return = torch.cat(
                                (features_to_return, features_of_that_class[:sample_per_class, :]), dim=0)
                            labels_to_return = torch.cat((labels_to_return, s.repeat(sample_per_class)),dim=0)

                    return features_to_return, labels_to_return
                else:
                    return torch.cuda.FloatTensor([]), torch.cuda.LongTensor([])


            # some of the following might be empty tensors if the specified number of
            # samples is zero :

            img_seen_feat,   img_seen_label   = sample_train_data_on_sample_per_class_basis(
                train_seen_feat,train_seen_label,self.img_seen_samples )

            img_unseen_feat, img_unseen_label = sample_train_data_on_sample_per_class_basis(
                train_unseen_feat, train_unseen_label, self.img_unseen_samples )

            att_unseen_feat, att_unseen_label = sample_train_data_on_sample_per_class_basis(
                    novelclass_aux_data,
                    novel_corresponding_labels,self.att_unseen_samples )

            att_seen_feat, att_seen_label = sample_train_data_on_sample_per_class_basis(
                seenclass_aux_data,
                seen_corresponding_labels, self.att_seen_samples)

            def convert_datapoints_to_z(features, encoder):
                if features.size(0) != 0:
                    mu_, logvar_ = encoder(features)
                    z = self.reparameterize(mu_, logvar_)
                    return z
                else:
                    return torch.cuda.FloatTensor([])

            z_seen_img   = convert_datapoints_to_z(img_seen_feat, self.encoder['resnet_features'])
            z_unseen_img = convert_datapoints_to_z(img_unseen_feat, self.encoder['resnet_features'])

            z_seen_att = convert_datapoints_to_z(att_seen_feat, self.encoder[self.auxiliary_data_source])
            z_unseen_att = convert_datapoints_to_z(att_unseen_feat, self.encoder[self.auxiliary_data_source])

            train_Z = [z_seen_img, z_unseen_img ,z_seen_att    ,z_unseen_att]
            train_L = [img_seen_label    , img_unseen_label,att_seen_label,att_unseen_label]

            # empty tensors are sorted out
            train_X = [train_Z[i] for i in range(len(train_Z)) if train_Z[i].size(0) != 0]
            train_Y = [train_L[i] for i in range(len(train_L)) if train_Z[i].size(0) != 0]

            train_X = torch.cat(train_X, dim=0)
            train_Y = torch.cat(train_Y, dim=0)

            att_unseen_from_att = self.decoder[self.auxiliary_data_source](z_unseen_att)
            if self.generalized:
                att_seen_from_img = self.decoder[self.auxiliary_data_source](z_seen_img)
                train_X_att = torch.cat((att_seen_from_img,att_unseen_from_att),0)
        ############################################################
        ##### initializing the classifier and train one epoch
        ############################################################
        torch.cuda.empty_cache()

        all_Y = torch.cat((test_novel_Y, test_seen_Y), 0)
        cls = classifier2.CLASSIFIER(clf, train_X, train_Y, all_X, all_Y,
                                    cls_seenclasses, cls_novelclasses,
                                    self.num_classes, self.device, self.lr_cls, 0.5, 80,
                                    self.classifier_batch_size,
                                    self.generalized)

        cls3 = classifier2.CLASSIFIER(clf3, train_X_att, train_Y, all_att, all_Y, cls_seenclasses, cls_novelclasses,
                                    self.num_novel_classes, self.device, self.lr_cls, 0.5, 50,
                                    self.classifier_batch_size,
                                    self.generalized)

        acc_seen, acc_unseen, acc_H = self.fuse_pred(cls.output, cls3.output, all_Y)
        print('GZSL Fused Acc_Seen:{:.1f}%, Acc_Unseen:{:.1f}%, Acc_H:{:.1f}%'.format(acc_seen,acc_unseen,acc_H))
        print('****************')
