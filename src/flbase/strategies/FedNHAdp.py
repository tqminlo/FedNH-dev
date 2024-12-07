from collections import OrderedDict, Counter
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch

try:
    import wandb
except ModuleNotFoundError:
    pass
from ..server import Server
from ..client import Client
from ..models.CNN import *
from ..models.MLP import *
from ..utils import setup_optimizer, linear_combination_state_dict, setup_seed
from ...utils import autoassign, save_to_pkl, access_last_added_element
from .FedUH import FedUHClient, FedUHServer
import math
import time
import torch.nn.functional as F


class FedNHAdpClient(FedUHClient):
    def __init__(self, criterion, trainset, testset, client_config, cid, device, **kwargs):
        super().__init__(criterion, trainset, testset,
                         client_config, cid, device, **kwargs)
        temp = [self.count_by_class[cls] if cls in self.count_by_class.keys() else 1e-12 for cls in
                range(client_config['num_classes'])]
        self.count_by_class_full = torch.tensor(temp).to(self.device)

    def _estimate_prototype(self):
        self.model.eval()
        self.model.return_embedding = True
        embedding_dim = self.model.prototype.shape[1]
        prototype = torch.zeros_like(self.model.prototype)
        with torch.no_grad():
            for i, (x, y) in enumerate(self.trainloader):
                # forward pass
                x, y = x.to(self.device), y.to(self.device)
                # feature_embedding is normalized
                feature_embedding, _ = self.model.forward(x)
                classes_shown_in_this_batch = torch.unique(y).cpu().numpy()
                for cls in classes_shown_in_this_batch:
                    mask = (y == cls)
                    feature_embedding_in_cls = torch.sum(feature_embedding[mask, :], dim=0)
                    prototype[cls] += feature_embedding_in_cls
        for cls in self.count_by_class.keys():
            # sample mean
            prototype[cls] /= self.count_by_class[cls]
            # normalization so that self.W.data is of the sampe scale as prototype_cls_norm
            prototype_cls_norm = torch.norm(prototype[cls]).clamp(min=1e-12)
            prototype[cls] = torch.div(prototype[cls], prototype_cls_norm)

            # reweight it for aggregartion
            prototype[cls] *= self.count_by_class[cls]

        self.model.return_embedding = False

        to_share = {'scaled_prototype': prototype, 'count_by_class_full': self.count_by_class_full}
        return to_share

    def _estimate_prototype_adv(self):
        self.model.eval()
        self.model.return_embedding = True
        embeddings = []
        labels = []
        weights = []
        prototype = torch.zeros_like(self.model.prototype)
        with torch.no_grad():
            for i, (x, y) in enumerate(self.trainloader):
                # forward pass
                x, y = x.to(self.device), y.to(self.device)
                # feature_embedding is normalized
                # use the latest prototype
                feature_embedding, logits = self.model.forward(x)
                prob_ = F.softmax(logits, dim=1)
                prob = torch.gather(prob_, dim=1, index=y.view(-1, 1))
                labels.append(y)
                weights.append(prob)
                embeddings.append(feature_embedding)
        self.model.return_embedding = False
        embeddings = torch.cat(embeddings, dim=0)
        labels = torch.cat(labels, dim=0)
        weights = torch.cat(weights, dim=0).view(-1, 1)
        for cls in self.count_by_class.keys():
            mask = (labels == cls)
            weights_in_cls = weights[mask, :]
            feature_embedding_in_cls = embeddings[mask, :]
            prototype[cls] = torch.sum(feature_embedding_in_cls * weights_in_cls, dim=0) / torch.sum(weights_in_cls)
            prototype_cls_norm = torch.norm(prototype[cls]).clamp(min=1e-12)
            prototype[cls] = torch.div(prototype[cls], prototype_cls_norm)

        # calculate predictive power
        to_share = {'adv_agg_prototype': prototype, 'count_by_class_full': self.count_by_class_full}
        return to_share

    def upload(self):
        if self.client_config['FedNH_client_adv_prototype_agg']:
            return self.new_state_dict, self._estimate_prototype_adv()
        else:
            return self.new_state_dict, self._estimate_prototype()


class FedNHAdpServer(FedUHServer):
    def __init__(self, server_config, clients_dict, exclude, **kwargs):
        super().__init__(server_config, clients_dict, exclude, **kwargs)
        if len(self.exclude_layer_keys) > 0:
            print(f"FedNHServer: the following keys will not be aggregated:\n ", self.exclude_layer_keys)

    def aggregate(self, client_uploads, round):
        server_lr = self.server_config['learning_rate'] * (self.server_config['lr_decay_per_round'] ** (round - 1))
        num_participants = len(client_uploads)
        update_direction_state_dict = None
        # agg weights for prototype
        cumsum_per_class = torch.zeros(self.server_config['num_classes']).to(self.clients_dict[0].device)
        agg_weights_vec_dict = {}
        with torch.no_grad():
            # for idx, (client_state_dict, prototype_dict) in enumerate(client_uploads):
            #     if self.server_config['FedNH_server_adv_prototype_agg'] == False:
            #         cumsum_per_class += prototype_dict['count_by_class_full']
            #     else:
            #         mu = prototype_dict['adv_agg_prototype']
            #         W = self.server_model_state_dict['prototype']
            #         agg_weights_vec_dict[idx] = torch.exp(torch.sum(W * mu, dim=1, keepdim=True))
            #     client_update = linear_combination_state_dict(client_state_dict,
            #                                                   self.server_model_state_dict,
            #                                                   1.0,
            #                                                   -1.0,
            #                                                   exclude=self.exclude_layer_keys
            #                                                   )
            #     if idx == 0:
            #         update_direction_state_dict = client_update
            #     else:
            #         update_direction_state_dict = linear_combination_state_dict(update_direction_state_dict,
            #                                                                     client_update,
            #                                                                     1.0,
            #                                                                     1.0,
            #                                                                     exclude=self.exclude_layer_keys
            #                                                                     )
            #
            # # new feature extractor
            # self.server_model_state_dict = linear_combination_state_dict(self.server_model_state_dict,
            #                                                              update_direction_state_dict,
            #                                                              1.0,
            #                                                              server_lr / num_participants,
            #                                                              exclude=self.exclude_layer_keys
            #                                                              )
            # print("---check server_lr: ", server_lr)
            # print("---check model weight: ", self.server_model_state_dict.keys())
            # print("---check exclude_layer_keys: ", self.exclude_layer_keys)

            gradients_list = []
            for idx, (client_state_dict, prototype_dict) in enumerate(client_uploads):
                gradient = []
                for state_key in client_state_dict.keys():
                    if state_key not in self.exclude_layer_keys:
                        g = client_state_dict[state_key] - self.server_model_state_dict[state_key]
                        gradient.append(g.cpu().numpy().flatten())
                gradient = np.concatenate(gradient, axis=0)
                gradients_list.append(gradient)
                # print("---check gradient shape : ", gradient.shape)

            weights = torch.tensor([1] * len(gradients_list))
            # print("---check weights : ", weights)
            weights = weights / torch.sum(weights)
            weights_2d = weights.unsqueeze(1)
            # print("---check weights_2d : ", weights_2d)
            gradients_list = torch.tensor(gradients_list)
            print("---check gradients_list : ", gradients_list.shape)
            gradient_global = gradients_list * weights_2d
            gradient_global = torch.sum(gradient_global, 0)
            cosin_similary_list = [torch.nn.CosineSimilarity(dim=0)(gradient_global, grad_k) for grad_k in
                                   gradients_list]
            thetas = torch.tensor([torch.arccos(cos).item() for cos in cosin_similary_list])

            # if round == 1:
            #     self.thetas = thetas
            # else:
            #     self.thetas = self.thetas * (round-1)/round + thetas * 1/round
            # print("---1--- thetas smooth:", self.thetas)

            def gompertz_func(theta, alpha=5.):
                return alpha * (1 - np.exp(-np.exp(-alpha * (theta - 1))))

            # re_weights = np.exp(gompertz_func(self.thetas, alpha=5.))
            re_weights = np.exp(gompertz_func(thetas, alpha=5.))
            # re_weights = gompertz_func(thetas, alpha=5.)
            # re_weights = np.power(gompertz_func(thetas, alpha=5.), 1/round)
            # re_weights = np.power(gompertz_func(thetas, alpha=5.), 1/8)
            re_weights = re_weights / torch.sum(re_weights)
            print("---2--- re_weights:", re_weights)
            assert torch.all(re_weights >= 0), "weights should be non-negative values"

            for idx, (client_state_dict, prototype_dict) in enumerate(client_uploads):
                if self.server_config['FedNH_server_adv_prototype_agg'] == False:
                    cumsum_per_class += prototype_dict['count_by_class_full']
                else:
                    mu = prototype_dict['adv_agg_prototype']
                    W = self.server_model_state_dict['prototype']
                    agg_weights_vec_dict[idx] = torch.exp(torch.sum(W * mu, dim=1, keepdim=True))

                for state_key in client_state_dict.keys():
                    if state_key not in self.exclude_layer_keys:
                        if idx == 0:
                            self.server_model_state_dict[state_key] = client_state_dict[state_key] * re_weights[idx]
                        else:
                            self.server_model_state_dict[state_key] += client_state_dict[state_key] * re_weights[idx]

            avg_prototype = torch.zeros_like(self.server_model_state_dict['prototype'])

            # if self.server_config['FedNH_server_adv_prototype_agg'] == False:
            #     # print("---check--- cumsum_per_class : ", cumsum_per_class.view(-1, 1))
            #     for _, prototype_dict in client_uploads:
            #         avg_prototype += prototype_dict['scaled_prototype'] / cumsum_per_class.view(-1, 1)
            #     print("---3--- cumsum_per_class:", cumsum_per_class)

            if self.server_config['FedNH_server_adv_prototype_agg'] == False:
                # print("---check--- cumsum_per_class : ", cumsum_per_class.view(-1, 1))
                for idx, (client_state_dict, prototype_dict) in enumerate(client_uploads):
                    avg_prototype += (prototype_dict['scaled_prototype'] / cumsum_per_class.view(-1, 1)) * re_weights[
                        idx]
                print("---3--- cumsum_per_class:", cumsum_per_class)
            else:
                m = self.server_model_state_dict['prototype'].shape[0]
                sum_of_weights = torch.zeros((m, 1)).to(avg_prototype.device)
                for idx, (_, prototype_dict) in enumerate(client_uploads):
                    sum_of_weights += agg_weights_vec_dict[idx]
                    avg_prototype += agg_weights_vec_dict[idx] * prototype_dict['adv_agg_prototype']
                avg_prototype /= sum_of_weights

            # normalize prototype
            avg_prototype = F.normalize(avg_prototype, dim=1)
            # update prototype with moving average
            weight = self.server_config['FedNH_smoothing']
            temp = weight * self.server_model_state_dict['prototype'] + (1 - weight) * avg_prototype
            # print('agg weight:', weight)
            # normalize prototype again
            self.server_model_state_dict['prototype'].copy_(F.normalize(temp, dim=1))

    def run(self, **kwargs):
        if self.server_config['use_tqdm']:
            round_iterator = tqdm(range(self.rounds + 1, self.server_config['num_rounds'] + 1), desc="Round Progress")
        else:
            round_iterator = range(self.rounds + 1, self.server_config['num_rounds'] + 1)
        # round index begin with 1
        for r in round_iterator:
            setup_seed(r + kwargs['global_seed'])
            selected_indices = self.select_clients(self.server_config['participate_ratio'])
            if self.server_config['drop_ratio'] > 0:
                # mimic the stragler issues; simply drop them
                self.active_clients_indicies = np.random.choice(selected_indices, int(
                    len(selected_indices) * (1 - self.server_config['drop_ratio'])), replace=False)
            else:
                self.active_clients_indicies = selected_indices
            # active clients download weights from the server
            tqdm.write(f"Round:{r} - Active clients:{self.active_clients_indicies}:")
            # self.active_clients_indicies = np.array([5, 15, 25, 35, 45, 55, 65, 76, 85, 95])
            # self.active_clients_indicies = np.arange(100)
            # print("*** CHECK active_clients_indicies : ", self.active_clients_indicies)
            for cid in self.active_clients_indicies:
                client = self.clients_dict[cid]
                client.set_params(self.server_model_state_dict, self.exclude_layer_keys)

            # clients perform local training
            train_start = time.time()
            client_uploads = []
            for cid in self.active_clients_indicies:
                client = self.clients_dict[cid]
                client.training(r, kwargs['num_epochs'])
                client_uploads.append(client.upload())
            train_time = time.time() - train_start
            print(f" Training time:{train_time:.3f} seconds")
            # collect training stats
            # average train loss and acc over active clients, where each client uses the latest local models
            self.collect_stats(stage="train", round=r, active_only=True)

            # get new server model
            # agg_start = time.time()
            self.aggregate(client_uploads, round=r)
            # agg_time = time.time() - agg_start
            # print(f" Aggregation time:{agg_time:.3f} seconds")
            # collect testing stats
            if (r - 1) % self.server_config['test_every'] == 0:
                test_start = time.time()
                self.testing(round=r, active_only=True)
                test_time = time.time() - test_start
                print(f" Testing time:{test_time:.3f} seconds")
                self.collect_stats(stage="test", round=r, active_only=True)
                print(" avg_test_acc:", self.gfl_test_acc_dict[r]['acc_by_criteria'])
                print(" pfl_avg_test_acc:", self.average_pfl_test_acc_dict[r])
                if len(self.gfl_test_acc_dict) >= 2:
                    current_key = r
                    if self.gfl_test_acc_dict[current_key]['acc_by_criteria']['uniform'] > best_test_acc:
                        best_test_acc = self.gfl_test_acc_dict[current_key]['acc_by_criteria']['uniform']
                        self.server_model_state_dict_best_so_far = deepcopy(self.server_model_state_dict)
                        tqdm.write(
                            f" Best test accuracy:{float(best_test_acc):5.3f}. Best server model is updatded and saved at {kwargs['filename']}!")
                        if 'filename' in kwargs:
                            torch.save(self.server_model_state_dict_best_so_far, kwargs['filename'])
                else:
                    best_test_acc = self.gfl_test_acc_dict[r]['acc_by_criteria']['uniform']
            # wandb monitoring
            if kwargs['use_wandb']:
                stats = {"avg_train_loss": self.average_train_loss_dict[r],
                         "avg_train_acc": self.average_train_acc_dict[r],
                         "gfl_test_acc_uniform": self.gfl_test_acc_dict[r]['acc_by_criteria']['uniform']
                         }

                for criteria in self.average_pfl_test_acc_dict[r].keys():
                    stats[f'pfl_test_acc_{criteria}'] = self.average_pfl_test_acc_dict[r][criteria]

                wandb.log(stats)