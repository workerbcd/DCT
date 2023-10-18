from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np

class CDRandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    The whole samples in a batch are from the same camera.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, datasets, batch_size, cameranum, classnum,drop_out=True):
        self.datas = datasets
        self.batch_size = batch_size

        self.index_dic = defaultdict(lambda: defaultdict(list)) #dict with dict value
        self.drop_out = drop_out
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        self.cameranum = cameranum
        self.classnum = classnum
        self.length = 0
        self.instance = cameranum+1
        self.num_per_batch = batch_size//self.instance
        assert self.num_per_batch>self.cameranum, \
            print("num of class per batch is {0},it must be larger then {1}".format(self.num_per_batch,self.cameranum))
        for index, (_, pid, camid) in enumerate(self.datas):
            self.index_dic[pid][camid].append(index)
            # self.length += 1

    def __iter__(self):
        """
        kind of balance the domains and classes,
        we do not consider if the two classes or domains have the same num
        """
        cluster_dict = defaultdict(list)
        camera_dict = defaultdict(list)
        # num_pids = copy.deepcopy(self.num_per_id)
        camera_list = list(range(self.cameranum))
        for i in range(len([*self.index_dic])):
            i_doms = copy.deepcopy(self.index_dic[i])
            flags = [1]*self.cameranum
            while sum(flags)>0:
                weights = [len(i_doms[j]) if flags[j]==1 else 1 for j in range(self.cameranum)]
                cameras = random.choices(camera_list,weights=weights,k=self.instance)
                while len(list(set(cameras)))<=1:
                    cameras = random.choices(camera_list, k=self.instance)
                indxs = []
                cluster_camera = []
                for c in cameras:
                    indx = random.choice(i_doms[c])
                    i_doms[c].remove(indx)
                    if len(i_doms[c])==0:
                        i_doms[c] = copy.deepcopy(self.index_dic[i][c])
                        flags[c] = 0
                        weights[c] = weights[c]/10.0
                    indxs.append(indx)
                    cluster_camera.append(c)
                random.shuffle(indxs)
                cluster_dict[i].append(indxs)
                camera_dict[i].append(cluster_camera)


        final_idxs = []
        flags = [1]*self.classnum
        pid_list = list(range(self.classnum))

        batch_idxs = copy.deepcopy(cluster_dict)
        batch_cameras = copy.deepcopy(camera_dict)
        pid_weights = [len(batch_idxs[j]) for j in range(self.classnum)]

        while sum(flags)>0:
            pid_weights = [len(batch_idxs[j]) if flags[j]==1 else min(pid_weights[j],1) for j in range(self.classnum)]
            selected_pid = random.choices(pid_list,weights=pid_weights,k=self.num_per_batch)
            unq_num = np.unique(selected_pid)
            while len(unq_num)<=1:
                selected_pid = random.choices(pid_list, weights=pid_weights, k=self.num_per_batch)
                unq_num = np.unique(selected_pid)
            t_final_idxs=[]
            camera_idxs = []
            for c in range(self.cameranum):
                camera_idxs.append([])
            for p in selected_pid:
                indxs = batch_idxs[p].pop(0)
                tcs = batch_cameras[p].pop(0)
                for ti in range(len(tcs)):
                    tc,tp = tcs[ti],selected_pid[ti]
                    camera_idxs[tc].append(tp)
                t_final_idxs.extend(indxs)
                if len(batch_idxs[p])==0:
                    batch_idxs[p] = copy.deepcopy(cluster_dict[p])
                    batch_cameras[p] = copy.deepcopy(camera_dict[p])
                    flags[p]=0
                    pid_weights[p] /= 10.0
            min_c_num = min([len(np.unique(camera_idxs[c])) for c in range(self.cameranum)])
            if min_c_num>1: final_idxs.extend(t_final_idxs)
            else:
                print(camera_idxs)
            # final_idxs.extend(t_final_idxs)
        if self.drop_out:
            final_length = (len(final_idxs)//self.batch_size)*self.batch_size
            final_idxs = final_idxs[:final_length]
        self.length=len(final_idxs)
        print(len(final_idxs),self.length)

        return iter(final_idxs)

    def __len__(self):
        return self.length

