import os
import os.path as osp
from scipy.io import loadmat
import cv2
import numpy as np
import json
from PIL import Image
import pickle
import re
import sys
#from numba import jit
from km import run_kuhn_munkres

from sklearn.metrics import average_precision_score
from sklearn.preprocessing import normalize

from iou_utils import get_max_iou

def compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter * 1.0 / union

def set_box_pid(boxes, box, pids, pid):
    for i in range(boxes.shape[0]):
        if np.all(boxes[i] == box):
            pids[i] = pid
            return
    print("Person: %s, box: %s cannot find in images." % (pid, box))

def image_path_at(data_path, image_index, i):
    image_path = osp.join(data_path, image_index[i])
    assert osp.isfile(image_path), "Path does not exist: %s" % image_path
    return image_path

def load_image_index(root_dir, db_name):
    """Load the image indexes for training / testing."""
    # Test images
    test = loadmat(osp.join(root_dir, "annotation", "pool.mat"))
    test = test["pool"].squeeze()
    test = [str(a[0]) for a in test]
    if db_name == "psdb_test":
        return test

    # All images
    all_imgs = loadmat(osp.join(root_dir, "annotation", "Images.mat"))
    all_imgs = all_imgs["Img"].squeeze()
    all_imgs = [str(a[0][0]) for a in all_imgs]

    # Training images = all images - test images
    train = list(set(all_imgs) - set(test))
    train.sort()
    return train

def _get_cam_id(im_name):
        match = re.search('c\d', im_name).group().replace('c', '')
        return int(match)

def load_probes(root):
    query_info = osp.join(root, 'query_info.txt')
    with open(query_info, 'r') as f:
        raw = f.readlines()

    probes = []
    for line in raw:
        linelist = line.split(' ')
        pid = int(linelist[0])
        x, y, w, h = float(linelist[1]), float(
            linelist[2]), float(linelist[3]), float(linelist[4])
        roi = np.array([x, y, x + w, y + h]).astype(np.int32)
        roi = np.clip(roi, 0, None)  # several coordinates are negative
        im_name = linelist[5][:-1] + '.jpg'
        probes.append({'im_name': im_name,
                        'boxes': roi[np.newaxis, :],
                        # Useless. Can be set to any value.
                        'gt_pids': np.array([pid]),
                        'flipped': False,
                        'cam_id': _get_cam_id(im_name)
                        })

    return probes

def gt_roidbs(root):

    imgs = loadmat(osp.join(root, 'frame_test.mat'))['img_index_test']
    imgs = [img[0][0] + '.jpg' for img in imgs]

    gt_roidb = []
    for im_name in imgs:
        anno_path = osp.join(root, 'annotations', im_name)
        anno = loadmat(anno_path)
        box_key = 'box_new'
        if box_key not in anno.keys():
            box_key = 'anno_file'
        if box_key not in anno.keys():
            box_key = 'anno_previous'

        rois = anno[box_key][:, 1:]
        ids = anno[box_key][:, 0]
        rois = np.clip(rois, 0, None)  # several coordinates are negative

        assert len(rois) == len(ids)

        rois[:, 2:] += rois[:, :2]
        # num_objs = len(rois)
        # overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # overlaps[:, 1] = 1.0
        # overlaps = csr_matrix(overlaps)
        gt_roidb.append({
            'im_name': im_name,
            'boxes': rois.astype(np.int32),
            'gt_pids': ids.astype(np.int32),
            'flipped': False,
            'cam_id': _get_cam_id(im_name)
            # 'gt_overlaps': overlaps
        })
    return gt_roidb

# @jit(forceobj=True)
def main(det_thresh=0.05, gallery_size=-1, ignore_cam_id=True, input_path=None, cbgm=False):

    # change here
    # results_path = './work_dirs/' + input_path
    data_root='./data/PRW-v16.04.20/'
    probe_set = load_probes(data_root)
    gallery_set = gt_roidbs(data_root)


    name_id = dict()
    for i, gallery in enumerate(gallery_set):
        name = gallery['im_name']
        name_id[name] = i
    # print(name_id)

    with open(input_path, 'rb') as fid:
        all_dets = pickle.load(fid)

    gallery_det, gallery_feat = [], []
    for det in all_dets:
        gallery_det.append(det[0][:, :5])
        if det[0].shape[0] > 0:
            feat = normalize(det[0][:, 5:], axis=1)
            # feat = det[0][:, 5:]
        else:
            feat = det[0][:, 5:]
        # feat = normalize(det[0][:, 5:], axis=1)
        gallery_feat.append(feat)

    probe_feat = []
    query_feat = []
    query_det = []

    for i in range(len(probe_set)):
        probe = probe_set[i]
        name = probe['im_name']
        query_gt_box = probe['boxes'][0]
        id = name_id[name]
        det0 = gallery_det[id]
        feat0 = gallery_feat[id]


        iou, iou_max, nmax = get_max_iou(det0, query_gt_box)
        if iou_max < 0.1:
            print("not detected", name, iou_max)

        feat = feat0[nmax]
        probe_feat.append(feat)



        inds = np.where(det0[:,-1] >= det_thresh)[0]
        flag = (inds != nmax)
        inds = inds[flag]
        if len(inds) > 0:
            detx = det0[inds]
            featx = feat0[inds]
            inds1 = np.argsort(detx[:, -1])[::-1]
            q_feat = np.concatenate(([feat], featx[inds1]), axis=0)
            q_det = np.concatenate(([det0[nmax]], detx[inds1]), axis=0)
            query_feat.append(q_feat)
            query_det.append(q_det)
        else:
            query_feat.append(feat.reshape(1,-1))
            query_det.append(det0[nmax].reshape(1,-1))

        # ids1 = (inds != nmax)
        # inds = np.append(nmax,inds[ids1])
        # query_feat.append()
        # query_det.append(det[inds])


    # gallery_det, gallery_feat = [], []
    # for det in all_dets:
        # det[0] = det[0][det[0][:, 4]>thresh]
        # gallery_det.append(det[0][:, :5])
        # if det[0].shape[0] > 0:
        #     feat = normalize(det[0][:, 5:], axis=1)
        # else:
        #     feat = det[0][:, 5:]
        # feat = normalize(det[0][:, 5:], axis=1)
        # gallery_feat.append(feat)

    search_performance_calc(gallery_set, probe_set, gallery_det, gallery_feat, probe_feat, query_feat, query_det, det_thresh, gallery_size, ignore_cam_id, cbgm)

# @jit(forceobj=True)
def search_performance_calc(gallery_set, probe_set,
                                gallery_det, gallery_feat, probe_feat, query_feat, query_det,
                                det_thresh=0.5, gallery_size=-1, ignore_cam_id=True, cbgm = False):

    assert len(gallery_set) == len(gallery_det)
    assert len(gallery_set) == len(gallery_feat)
    assert len(probe_set) == len(probe_feat)

    gt_roidb = gallery_set
    name_to_det_feat = {}
    for gt, det, feat in zip(gt_roidb, gallery_det, gallery_feat):
        name = gt['im_name']
        pids = gt['gt_pids']
        cam_id = gt['cam_id']
        scores = det[:, 4].ravel()
        inds = np.where(scores >= det_thresh)[0]
        # print(len(inds),len(scores))
        if len(inds) > 0:
            name_to_det_feat[name] = (det[inds], feat[inds], pids, cam_id)

    aps = []
    accs = []
    topk = [1, 5, 10]
    # ret = {'image_root': gallery_set.data_path, 'results': []}
    for i in range(len(probe_set)):
        y_true, y_score = [], []
        imgs, rois = [], []
        count_gt, count_tp = 0, 0

        feat_p = probe_feat[i].ravel()

        probe_imname = probe_set[i]['im_name']
        probe_roi = probe_set[i]['boxes'].ravel()
        probe_pid = probe_set[i]['gt_pids']
        probe_cam = probe_set[i]['cam_id']

        # Find all occurence of this probe
        gallery_imgs = []
        for x in gt_roidb:
            if probe_pid in x['gt_pids'] and x['im_name'] != probe_imname:
                gallery_imgs.append(x)

        probe_gts = {}
        for item in gallery_imgs:
            probe_gts[item['im_name']] = \
                item['boxes'][item['gt_pids'] == probe_pid]

        # Construct gallery set for this probe
        if ignore_cam_id:
            gallery_imgs = []
            for x in gt_roidb:
                if x['im_name'] != probe_imname:
                    gallery_imgs.append(x)
        else:
            gallery_imgs = []
            for x in gt_roidb:
                if x['im_name'] != probe_imname and x['cam_id'] != probe_cam:
                    gallery_imgs.append(x)

        # # 1. Go through all gallery samples
        # for item in testset.targets_db:
        # Gothrough the selected gallery
        # for item in gallery_imgs:
        #     gallery_imname = item['im_name']
        #     # some contain the probe (gt not empty), some not
        #     count_gt += (gallery_imname in probe_gts)
        #     # compute distance between probe and gallery dets
        #     if gallery_imname not in name_to_det_feat:
        #         continue
        #     det, feat_g, _, _ = name_to_det_feat[gallery_imname]
        #
        #     # get L2-normalized feature matrix NxD
        #     assert feat_g.size == np.prod(feat_g.shape[:2])
        #     feat_g = feat_g.reshape(feat_g.shape[:2])
        #     # compute cosine similarities
        #     sim = feat_g.dot(feat_p).ravel()
        #     # assign label for each det
        #     label = np.zeros(len(sim), dtype=np.int32)
        #     if gallery_imname in probe_gts:
        #         gt = probe_gts[gallery_imname].ravel()
        #         w, h = gt[2] - gt[0], gt[3] - gt[1]
        #         iou_thresh = min(0.5, (w * h * 1.0) /
        #                             ((w + 10) * (h + 10)))
        #         #iou_thresh = min(0.3, (w * h * 1.0) /
        #         #                    ((w + 10) * (h + 10)))
        #         inds = np.argsort(sim)[::-1]
        #         sim = sim[inds]
        #         det = det[inds]
        #         # only set the first matched det as true positive
        #         for j, roi in enumerate(det[:, :4]):
        #             if compute_iou(roi, gt) >= iou_thresh:
        #                 label[j] = 1
        #                 count_tp += 1
        #                 break
        #     y_true.extend(list(label))
        #     y_score.extend(list(sim))
        #     imgs.extend([gallery_imname] * len(sim))
        #     rois.extend(list(det))

        name2sim = {}
        sims = []
        imgs_cbgm = []
        for item in gallery_imgs:
            gallery_imname = item['im_name']
            # some contain the probe (gt not empty), some not
            count_gt += (gallery_imname in probe_gts)
            # compute distance between probe and gallery dets
            if gallery_imname not in name_to_det_feat:
                continue
            det, feat_g, _, _ = name_to_det_feat[gallery_imname]
            # get L2-normalized feature matrix NxD
            assert feat_g.size == np.prod(feat_g.shape[:2])
            feat_g = feat_g.reshape(feat_g.shape[:2])
            # compute cosine similarities
            sim = feat_g.dot(feat_p).ravel()
            if gallery_imname in name2sim:
                continue
            name2sim[gallery_imname] = sim
            sims.extend(list(sim))
            imgs_cbgm.extend([gallery_imname]*len(sim))


        k1 = 30
        k2 = 3
        if cbgm:
            sims = np.array(sims)
            imgs_cbgm = np.array(imgs_cbgm)
            inds = np.argsort(sims)[-k1:]
            imgs_cbgm = set(imgs_cbgm[inds])
            for img in imgs_cbgm:
                sim = name2sim[img]
                det, feat_g, _, _ = name_to_det_feat[img]
                qboxes = query_det[i][:k2]
                qfeats = query_feat[i][:k2]
                # print(i,probe_roi,qboxes[0])
                # assert (
                #     probe_roi - qboxes[0][:4]
                # ).sum() <= 0.001, "query_roi must be the first one in pboxes"


                graph = []
                for indx_i, pfeat in enumerate(qfeats):
                    for indx_j, gfeat in enumerate(feat_g):
                        graph.append((indx_i, indx_j, (pfeat * gfeat).sum()))
                km_res, max_val = run_kuhn_munkres(graph)

                for indx_i, indx_j, _ in km_res:
                    if indx_i == 0:
                        sim[indx_j] = (max_val+sim[indx_j])/2#max_val
                        break
        for gallery_imname, sim in name2sim.items():
            det, feat_g, _, _ = name_to_det_feat[gallery_imname]
            # assign label for each det
            label = np.zeros(len(sim), dtype=np.int32)
            if gallery_imname in probe_gts:
                gt = probe_gts[gallery_imname].ravel()
                w, h = gt[2] - gt[0], gt[3] - gt[1]
                iou_thresh = min(0.5, (w * h * 1.0) / ((w + 10) * (h + 10)))
                inds = np.argsort(sim)[::-1]
                sim = sim[inds]
                det = det[inds]
                # only set the first matched det as true positive
                for j, roi in enumerate(det[:, :4]):
                    if compute_iou(roi, gt) >= iou_thresh:
                        label[j] = 1
                        count_tp += 1
                        break
            y_true.extend(list(label))
            y_score.extend(list(sim))
            imgs.extend([gallery_imname] * len(sim))
            rois.extend(list(det))


        # 2. Compute AP for this probe (need to scale by recall rate)
        y_score = np.asarray(y_score)
        y_true = np.asarray(y_true)
        assert count_tp <= count_gt
        recall_rate = count_tp * 1.0 / count_gt
        ap = 0 if count_tp == 0 else \
            average_precision_score(y_true, y_score) * recall_rate
        aps.append(ap)
        inds = np.argsort(y_score)[::-1]
        y_score = y_score[inds]
        y_true = y_true[inds]
        accs.append([min(1, sum(y_true[:k])) for k in topk])
        # # 4. Save result for JSON dump
        # new_entry = {'probe_img': str(probe_imname),
        #                 'probe_roi': map(float, list(probe_roi.squeeze())),
        #                 'probe_gt': probe_gts,
        #                 'gallery': []}
        # # only save top-10 predictions
        # for k in range(10):
        #     new_entry['gallery'].append({
        #         'img': str(imgs[inds[k]]),
        #         'roi': map(float, list(rois[inds[k]])),
        #         'score': float(y_score[k]),
        #         'correct': int(y_true[k]),
        #     })
        # ret['results'].append(new_entry)

    print('search ranking:')
    mAP = np.mean(aps)
    print('  mAP = {:.2%}'.format(mAP))
    accs = np.mean(accs, axis=0)
    for i, k in enumerate(topk):
        print('  top-{:2d} = {:.2%}'.format(k, accs[i]))




if __name__ == "__main__":
    # for t in [0.05, 0.15, 0.25, 0.3, 0.35, 0.4]:
    #     print('---------')
    #     print(t)
    #     main(det_thresh=t, input_path=sys.argv[1])


    main(det_thresh=0.3, input_path=sys.argv[1],cbgm=True)
