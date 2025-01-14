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
from sklearn.utils import shuffle


class FedNHCS6Client(FedUHClient):
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


class FedNHCS6Server(FedUHServer):
    def __init__(self, server_config, clients_dict, exclude, **kwargs):
        super().__init__(server_config, clients_dict, exclude, **kwargs)
        if len(self.exclude_layer_keys) > 0:
            print(f"FedNHServer: the following keys will not be aggregated:\n ", self.exclude_layer_keys)
        # self.top10_first10round = np.zeros(shape=(10,))
        # self.the_rest = np.zeros(shape=(90,))
        self.top10_first10round = np.array([0, 9, 17, 26, 29, 31, 60, 61, 73, 81])
        self.sort_client = np.zeros(shape=(10, 10), dtype=int)

    def aggregate(self, client_uploads, round):
        server_lr = self.server_config['learning_rate'] * (self.server_config['lr_decay_per_round'] ** (round - 1))
        num_participants = len(client_uploads)
        update_direction_state_dict = None
        # agg weights for prototype
        cumsum_per_class = torch.zeros(self.server_config['num_classes']).to(self.clients_dict[0].device)
        agg_weights_vec_dict = {}

        if round <= 10:
            acc_list = []
            score_list = []
            for idx, (client_state_dict, prototype_dict) in enumerate(client_uploads):
                self.server_side_client.set_params(client_state_dict, self.exclude_layer_keys)
                self.server_side_client.testing(round, testloader=None)  # use global testdataset
                count_per_class_test = Counter(self.server_side_client.testloader.dataset.targets.numpy())
                num_classes = self.server_side_client.client_config['num_classes']
                count_per_class_test = torch.tensor([count_per_class_test[cls] * 1.0 for cls in range(num_classes)])
                # print("---check count_by_class_testset : ", count_per_class_test)
                count_per_class_client = prototype_dict['count_by_class_full']
                print("---check count_by_class_client : ", count_per_class_client)
                num_data_in_client = torch.sum(count_per_class_client)
                print("---check num_data_in_client : ", num_data_in_client)
                self.gfl_test_acc_dict[round] = self.server_side_client.test_acc_dict[round]
                acc = self.gfl_test_acc_dict[round]['acc_by_criteria']["uniform"]
                acc_per_class = self.gfl_test_acc_dict[round]['correct_per_class']
                print("---check acc_per_class : ", acc_per_class)

                score = torch.prod(np.exp(acc_per_class / count_per_class_test)) * num_data_in_client
                # acc_list.append(acc)
                score_list.append(score.cpu().numpy())

            # print("---check acc_list : ", acc_list)
            print("---check score_list : ", score_list)

            sorted_idx = [x for _, x in sorted(zip(score_list, self.active_clients_indicies), reverse=True)]
            sorted_idx = np.array(sorted_idx)
            print("---check sorted_idx : ", sorted_idx)

            # Calculate gradient list
            gradients_list = []
            for idx, (client_state_dict, prototype_dict) in enumerate(client_uploads):
                gradient = []
                for state_key in client_state_dict.keys():
                    if state_key not in self.exclude_layer_keys:
                        g = client_state_dict[state_key] - self.server_model_state_dict[state_key]
                        gradient.append(g.cpu().numpy().flatten())
                gradient = np.concatenate(gradient, axis=0)
                gradients_list.append(gradient)
            weights = torch.tensor([1] * 10)
            weights = weights / torch.sum(weights)
            weights_2d = weights.unsqueeze(1)
            gradients_list = torch.tensor(gradients_list)
            gradient_global = gradients_list * weights_2d
            gradient_global = torch.sum(gradient_global, 0)
            cosin_similary_list = [torch.nn.CosineSimilarity(dim=0)(gradient_global, grad_k) for grad_k in
                                   gradients_list]
            print("---check cosin_similary_list : ", cosin_similary_list)
            sorted_idx = np.array(
                [x for _, x in sorted(zip(cosin_similary_list, self.active_clients_indicies), reverse=True)])
            print("---check sorted_idx_grad : ", sorted_idx)

            # Sort
            score_list = torch.tensor(np.array(score_list))
            score_final = [score_list[i] * cosin_similary_list[i] for i in range(len(score_list))]
            sorted_idx = np.array([x for _, x in sorted(zip(score_final, self.active_clients_indicies), reverse=True)])
            print("---check sorted_idx_score : ", sorted_idx)
            self.sort_client[round-1] = sorted_idx

        if round == 10:
            self.sort_client = self.sort_client.transpose()
            print("---check self.sort_client : ", self.sort_client)

        with torch.no_grad():
            for idx, (client_state_dict, prototype_dict) in enumerate(client_uploads):
                if self.server_config['FedNH_server_adv_prototype_agg'] == False:
                    cumsum_per_class += prototype_dict['count_by_class_full']
                else:
                    mu = prototype_dict['adv_agg_prototype']
                    W = self.server_model_state_dict['prototype']
                    agg_weights_vec_dict[idx] = torch.exp(torch.sum(W * mu, dim=1, keepdim=True))
                client_update = linear_combination_state_dict(client_state_dict,
                                                              self.server_model_state_dict,
                                                              1.0,
                                                              -1.0,
                                                              exclude=self.exclude_layer_keys
                                                              )
                if idx == 0:
                    update_direction_state_dict = client_update
                else:
                    update_direction_state_dict = linear_combination_state_dict(update_direction_state_dict,
                                                                                client_update,
                                                                                1.0,
                                                                                1.0,
                                                                                exclude=self.exclude_layer_keys
                                                                                )
            # new feature extractor
            self.server_model_state_dict = linear_combination_state_dict(self.server_model_state_dict,
                                                                         update_direction_state_dict,
                                                                         1.0,
                                                                         server_lr / num_participants,
                                                                         exclude=self.exclude_layer_keys
                                                                         )
            avg_prototype = torch.zeros_like(self.server_model_state_dict['prototype'])
            if self.server_config['FedNH_server_adv_prototype_agg'] == False:
                for _, prototype_dict in client_uploads:
                    avg_prototype += prototype_dict['scaled_prototype'] / cumsum_per_class.view(-1, 1)
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
            ####### new CS here :
            if r <= 10:
                self.active_clients_indicies = np.arange((r - 1) * 10, r * 10)
            elif 11 <= r <= 65:
                # setup_seed(r + kwargs['global_seed'])
                k = int(np.abs(np.sqrt(2*(r-10)+1/8) - 1/2))
                residual = int((r-10) - k*(k+1)/2)
                self.active_clients_indicies = self.sort_client[residual-1] if residual > 0 else self.sort_client[k-1]
            else:
                setup_seed(r + kwargs['global_seed'])
                selected_indices = self.select_clients(self.server_config['participate_ratio'])
                if self.server_config['drop_ratio'] > 0:
                    # mimic the stragler issues; simply drop them
                    self.active_clients_indicies = np.random.choice(selected_indices, int(
                        len(selected_indices) * (1 - self.server_config['drop_ratio'])), replace=False)
                else:
                    self.active_clients_indicies = selected_indices
                ######

            # active clients download weights from the server
            tqdm.write(f"Round:{r} - Active clients:{self.active_clients_indicies}:")
            for cid in self.active_clients_indicies:
                client = self.clients_dict[cid]
                client.set_params(self.server_model_state_dict, self.exclude_layer_keys)

            # clients perform local training
            train_start = time.time()
            client_uploads = []
            for cid in self.active_clients_indicies:
                client = self.clients_dict[cid]
                client.training(r, client.client_config['num_epochs'])
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
