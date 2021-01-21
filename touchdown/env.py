import numpy as np
import glob, re, random
from base_navigator import BaseNavigator
from utils import load_datasets, print_progress, load_nav_graph, input_img, random_list
import networkx as nx
import torch
import torch.nn.functional as F
from pyxdameraulevenshtein import damerau_levenshtein_distance as edit_dis


_SUCCESS_THRESHOLD = 2


def load_features(feature_store):
    feature = {} 
    if feature_store:
        imgs = glob.glob(feature_store+"/*.npy")
        print("=================================")
        print("=====Loading image features======")
        for img in imgs:
            feature[re.split('[/.]', img)[-2]] = np.load(img)
    return feature, (464, 100)


class EnvBatch:
    def __init__(self, opts, features, img_size, batch_size=64, name=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.opts = opts
        self.name = name
        self.features = features
        self.image_w, self.image_h = img_size
        self.navs = []
        print("=====Initializing %s navigators=====" % self.name)
        for i in range(batch_size):  # tqdm(range(batch_size)):
            nav = BaseNavigator(self.opts)
            self.navs.append(nav)
        print("=====================================")

    def newEpisodes(self, panoIds, headings):
        """ Iteratively initialize the simulators for # of batchsize"""
        for i, (panoId, heading) in enumerate(zip(panoIds, headings)):
            self.navs[i].graph_state = (panoId, heading)
            
    def roll_img(self, img, pano, heading):
        shift_angle = self.navs[0].graph.nodes[pano].pano_yaw_angle - heading
        width = img.shape[1]
        shift = int(width * shift_angle / 360)
        img = np.roll(img, shift, axis=1)  # like 'abcd' -> 'bcda'
        return img
            
    def _get_imgs(self, batch_size):
        imgs = []
        for i in range(batch_size):
            nav = self.navs[i]
            pano, heading = nav.graph_state
            if self.features:
                img = self.roll_img(self.features[pano], pano, heading)
            else:
                img = input_img(pano, self.opts.img_feat_dir)
                img = self.roll_img(img, pano, heading)
            img = img[:, 182:282, :].squeeze()  # 100*100 at the center from 464*100
            img = img[np.newaxis, ...]  # shape: (1, 100, 100)
            imgs.append(img)
        imgs = np.array(imgs)
        return torch.from_numpy(imgs).to(self.device)
    
    def _get_gt_action(self, batch, graph, is_test):
        gt_action = []
        for i, item in enumerate(batch):
            nav = self.navs[i]
            panoid, heading = nav.graph_state
            if is_test:
                goal_panoid = item['main_pano']
                gt_path = nx.dijkstra_path(graph, panoid, goal_panoid)
                if len(gt_path) < 2:
                    gt_action.append(3)
                    continue
                gt_next_panoid = gt_path[1]
            else:
                gt_path = batch[i]['route_panoids']
                pano_index = gt_path.index(panoid)
                if pano_index < len(gt_path) - 1:
                    gt_next_panoid = gt_path[pano_index + 1]
                else:
                    gt_action.append(3)  # STOP
                    continue
            pano_neighbors = nav.graph.nodes[panoid].neighbors
            neighbors_id = [neighbor.panoid for neighbor in pano_neighbors.values()]
            gt_next_heading = list(pano_neighbors.keys())[neighbors_id.index(gt_next_panoid)]
            delta_heading = (gt_next_heading - heading) % 360
            if delta_heading == 0:
                gt_action.append(0)  # FORWARD
            elif delta_heading < 180:
                gt_action.append(2)  # RIGHT
            else:
                gt_action.append(1)  # LEFT
                
        gt_action = np.array(gt_action)
        return torch.from_numpy(gt_action).long().to(self.device)
    
    def _action_select(self, a_prob, ended, num_act_nav, trajs, total_steps, batch):
        """Called during testing."""
        a = []
        action_list = ["forward", "left", "right", "stop"]
        for i in range(len(batch)):
            nav = self.navs[i]
            if ended[i].item():
                a.append([3])
                continue
            action_index = a_prob[i].argmax()
            action = action_list[action_index]
            if action == "stop":
                ended[i] = 1
                num_act_nav[0] -= 1
            nav.step(action)
            a.append([action_list.index(action)])
            if not nav.prev_graph_state[0] == nav.graph_state[0]:
                new_pano, _ = nav.graph_state
                trajs[i].append(new_pano)
            total_steps[0] += 1
        return torch.LongTensor(a).to(self.device)

    def cal_cls(self, graph, traj, gt_traj):
        PC = np.mean(np.exp([-np.min(
                [nx.dijkstra_path_length(graph, traj_point, gt_traj_point)
                for traj_point in traj])
                for gt_traj_point in gt_traj]))
        EPL = PC * len(gt_traj)
        LS = EPL / (EPL + np.abs(EPL - len(traj)))
        return LS * PC  
    
    def cal_dtw(self, graph, prediction, reference, success):
        dtw_matrix = np.inf * np.ones((len(prediction) + 1, len(reference) + 1))
        dtw_matrix[0][0] = 0
        for i in range(1, len(prediction)+1):
            for j in range(1, len(reference)+1):
                best_previous_cost = min(
                    dtw_matrix[i-1][j], dtw_matrix[i][j-1], dtw_matrix[i-1][j-1])
                cost = nx.dijkstra_path_length(graph, prediction[i-1], reference[j-1])
                dtw_matrix[i][j] = cost + best_previous_cost
        dtw = dtw_matrix[len(prediction)][len(reference)]
        dtw_group = [dtw]
        ndtw = np.exp(-dtw/(_SUCCESS_THRESHOLD * np.sqrt(len(reference))))
        dtw_group += [ndtw, success * ndtw]
        return dtw_group
                    
    def _eva_metrics(self, trajs, batch, graph, metrics):
        for i, item in enumerate(batch):
            success = 0
            traj = trajs[i]
            gt_traj = item["route_panoids"]
            ed = edit_dis(traj, gt_traj)
            ed = 1 - ed / max(len(traj), len(gt_traj))
            target_list = list(nx.all_neighbors(graph, gt_traj[-1])) + [gt_traj[-1]]
            if traj[-1] in target_list:
                success = 1
                metrics[0] += 1 
                metrics[2] += ed
            metrics[1] += nx.dijkstra_path_length(graph, traj[-1], gt_traj[-1])
            if self.opts.CLS:
                metrics[3] += self.cal_cls(graph, traj, gt_traj)
            if self.opts.DTW:
                dtw_group = self.cal_dtw(graph, traj, gt_traj, success)
                for j in range(-3, 0):
                    metrics[j] += dtw_group[j]

    def action_step(self, target, ended, num_act_nav, trajs, total_steps):
        action_list = ["forward", "left", "right", "stop"]
        for i in range(len(ended)):
            nav = self.navs[i]
            if ended[i].item():
                continue
            action = action_list[target[i]]
            if action == "stop":
                ended[i] = 1
                num_act_nav[0] -= 1
            nav.step(action)
            if not nav.prev_graph_state[0] == nav.graph_state[0]:
                new_pano, _ = nav.graph_state
                trajs[i].append(new_pano)
            total_steps[0] += 1


class TouchdownBatch:
    def __init__(self, opts, features, img_size, batch_size=64, seed=10, splits=["train"],
                 tokenizer=None, name=None):
        self.env = EnvBatch(opts, features, img_size, batch_size, name)
        self.data = []
        self.opts = opts
        
        json_data = load_datasets(splits, opts)
        total_length = len(json_data)
        
        for i, item in enumerate(json_data):
            new_item = dict(item)
            instr = new_item["navigation_text"]
            if self.opts.model == 'vlntrans':
                new_item["encoder_input"] = tokenizer.encode_text(text_a=instr, text_b=None, max_seq_length=self.opts.max_instr_len)
            else:
                new_item["instr_encoding"] = tokenizer.encode_sentence(instr)
            self.data.append(new_item)
        
        self.seed = seed
        random.seed(self.seed)
        random.shuffle(self.data)
        self.ix = 0
        self.batch_size = batch_size
        self.splits = splits
        self._load_nav_graph()

    def _load_nav_graph(self):
        self.graph = load_nav_graph(self.opts)
        print("Loading navigation graph done.")
        
    def _next_minibatch(self):
        batch = self.data[self.ix:self.ix+self.batch_size]
        if len(batch) < self.batch_size:
            random.shuffle(self.data)
        else:
            self.ix += self.batch_size
        self.batch = batch
        
    def get_imgs(self):
        return self.env._get_imgs(len(self.batch))

    def reset(self, print_info=False):
        self._next_minibatch()
        panoIds = []
        headings = []
        trajs = []
        if print_info:
            print(self.batch[0]["navigation_text"])
        for item in self.batch:
            panoIds.append(item["route_panoids"][0])
            headings.append(0)
            trajs.append([panoIds[-1]])
            
        self.env.newEpisodes(panoIds, headings)
        
        return trajs  # returned a batch of the first panoid for each route_panoids
        
    def get_gt_action(self, is_test):
        return self.env._get_gt_action(self.batch, self.graph, is_test)

    def action_select(self, a_t, ended, num_act_nav, trajs, total_steps):
        return self.env._action_select(a_t, ended, num_act_nav, trajs, total_steps, self.batch)
            
    def reset_epoch(self):
        self.ix = 0
            
    def eva_metrics(self, trajs, metrics):
        self.env._eva_metrics(trajs, self.batch, self.graph, metrics)
