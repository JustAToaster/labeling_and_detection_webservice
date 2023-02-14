import sys

import numpy as np

def xywh2xyxy(box):
    x, y, w_half, h_half = box['centerX'], box['centerY'], box['boxWidth']/2, box['boxHeight']/2
    return [x - w_half, y - h_half, x + w_half, y + h_half]

def bb_intersection_over_union(boxA_xywh, boxB_xywh):
    # Convert from xywh to xyxy
    boxA = xywh2xyxy(boxA_xywh)
    boxB = xywh2xyxy(boxB_xywh)

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def calculate_ap_11_point_interp(rec, prec, recall_vals=11):
    mrec = []
    # mrec.append(0)
    [mrec.append(e) for e in rec]
    # mrec.append(1)
    mpre = []
    # mpre.append(0)
    [mpre.append(e) for e in prec]
    # mpre.append(0)
    recallValues = np.linspace(0, 1, recall_vals)
    recallValues = list(recallValues[::-1])
    rhoInterp = []
    recallValid = []
    # For each recallValues (0, 0.1, 0.2, ... , 1)
    for r in recallValues:
        # Obtain all recall values higher or equal than r
        argGreaterRecalls = np.argwhere(mrec[:] >= r)
        pmax = 0
        # If there are recalls above r
        if argGreaterRecalls.size != 0:
            pmax = max(mpre[argGreaterRecalls.min():])
        recallValid.append(r)
        rhoInterp.append(pmax)
    # By definition AP = sum(max(precision whose recall is above r))/11
    ap = sum(rhoInterp) / len(recallValues)
    # Generating values for the plot
    rvals = []
    rvals.append(recallValid[0])
    [rvals.append(e) for e in recallValid]
    rvals.append(0)
    pvals = []
    pvals.append(0)
    [pvals.append(e) for e in rhoInterp]
    pvals.append(0)
    # rhoInterp = rhoInterp[::-1]
    cc = []
    for i in range(len(rvals)):
        p = (rvals[i], pvals[i - 1])
        if p not in cc:
            cc.append(p)
        p = (rvals[i], pvals[i])
        if p not in cc:
            cc.append(p)
    recallValues = [i[0] for i in cc]
    rhoInterp = [i[1] for i in cc]
    return [ap, rhoInterp, recallValues, None]

def ap_for_each_class(gt_boxes, det_boxes, iou_threshold=0.5):
    """Args:
        boundingboxes: list of dictionaries of bounding boxes in the xywh format
        iou_threshold: IOU threshold indicating which detections will be considered TP or FP"""
    aps_dict = {}
    # Get classes of all bounding boxes separating them by classes
    gt_classes_only = []
    classes_bbs = {}
    for bb in gt_boxes:
        c = bb['class_index']
        gt_classes_only.append(c)
        classes_bbs.setdefault(c, {'gt': [], 'det': []})
        classes_bbs[c]['gt'].append(bb)
    gt_classes_only = list(set(gt_classes_only))
    for bb in det_boxes:
        c = bb['class_index']
        classes_bbs.setdefault(c, {'gt': [], 'det': []})
        classes_bbs[c]['det'].append(bb)

    # Precision x Recall is obtained individually by each class
    for c, v in classes_bbs.items():
        # Report results only in the classes that are in the GT
        if c not in gt_classes_only:
            continue
        npos = len(v['gt'])
        # sort detections by decreasing confidence
        dects = [a for a in sorted(v['det'], key=lambda bb: bb['confidence'], reverse=True)]
        TP = np.zeros(len(dects))
        FP = np.zeros(len(dects))
        # create boolean list for gt detections
        detected_gt = [0] * len(gt_boxes)

        # Loop through detections
        for idx_det, det in enumerate(dects):

            # Find ground truth
            gt = [gt for gt in classes_bbs[c]['gt']]
            # Get the maximum iou among all detections in the image
            iouMax = sys.float_info.min
            # Given the detection set, find ground-truth with the highest iou
            for j, g in enumerate(gt):
                iou = bb_intersection_over_union(det, g)
                if iou > iouMax:
                    iouMax = iou
                    id_match_gt = j
            # Assign detection as TP or FP
            if iouMax >= iou_threshold:
                # gt was not matched with any detection
                if detected_gt[id_match_gt] == 0:
                    TP[idx_det] = 1  # detection is set as true positive
                    detected_gt[id_match_gt] = 1  # set flag to identify gt as already 'matched'
                    # print("TP")
                else:
                    FP[idx_det] = 1  # detection is set as false positive
                    # print("FP")
            # - A detected "cat" is overlaped with a GT "cat" with IOU >= iou_threshold.
            else:
                FP[idx_det] = 1  # detection is set as false positive
                # print("FP")
        # compute precision, recall and average precision
        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        rec = acc_TP / npos
        prec = np.divide(acc_TP, (acc_FP + acc_TP))
        [ap, mpre, mrec, _] = calculate_ap_11_point_interp(rec, prec)
        # add class result in the dictionary to be returned
        aps_dict[c] = ap
        #ret[c] = {
        #    'precision': prec,
        #    'recall': rec,
        #    'AP': ap,
        #    'interpolated precision': mpre,
        #    'interpolated recall': mrec,
        #    'total positives': npos,
        #    'total TP': np.sum(TP),
        #    'total FP': np.sum(FP),
        #    'iou': iou_threshold
        #}
    # For mAP, only the classes in the gt set should be considered
    # mAP = sum([v['AP'] for k, v in ret.items() if k in gt_classes_only]) / len(gt_classes_only)
    # return {'per_class': ret, 'mAP': mAP}
    return aps_dict

def iou_dist_each_class(predicted_boxes, custom_boxes):
    """Args:
        boundingboxes: list of dictionaries of bounding boxes in the xywh format"""
    iou_sum = {}
    iou_distances = {}
    # Get classes of all bounding boxes separating them by classes
    pred_classes_only = []
    classes_bbs = {}
    for bb in predicted_boxes:
        c = bb['class_index']
        pred_classes_only.append(c)
        classes_bbs.setdefault(c, {'pred': [], 'custom': []})
        classes_bbs[c]['pred'].append(bb)
    pred_classes_only = list(set(pred_classes_only))
    for bb in custom_boxes:
        c = bb['class_index']
        classes_bbs.setdefault(c, {'pred': [], 'custom': []})
        classes_bbs[c]['custom'].append(bb)

    for c in pred_classes_only:
        iou_sum[c] = 0.0

    for pred_box in predicted_boxes:
        iouMax = 0.0
        c = pred_box['class_index']
        # Given the detection with class c, find custom box with class c and the highest iou with the detection
        for custom_box in classes_bbs[c]['custom']:
            iou = bb_intersection_over_union(pred_box, custom_box)
            if iou > iouMax:
                iouMax = iou
        iou_sum[c] += iouMax
    for c in pred_classes_only:
        iou_distances[c] = 1.0 - iou_sum[c]/len(classes_bbs[c]['pred'])
    return iou_distances

def weighted_geometric_mean(validation_APs, distances, validation_weight=0.6, distance_weight=0.4):
    weighted_products = np.power(validation_APs, validation_weight) * np.power(distances, distance_weight)
    weighted_geometric_mean = np.power(weighted_products, 1/(validation_weight+distance_weight))
    return weighted_geometric_mean

# This function combines class scores so that scores closer to 1 are more important, still resulting in an overall higher score
def combine_class_scores(class_scores, weight=3):
    print(np.power(class_scores, weight))
    return np.sum(np.power(class_scores, weight)) / (np.sum(np.power(class_scores, weight - 1)) + 1e-5)

# If validation AP for a certain class is high, but the customization is a lot different than the prediction, it might be a malicious request
def customization_score(predicted_labels, customized_labels, val_AP_classes):
    
    # If there is no validation data or there were no boxes in both the prediction and the customization, there is no data to go by, so we have no choice but to trust it
    if not val_AP_classes or (not predicted_labels and not customized_labels):
        return 0.0
    
    #cust_AP_classes = ap_for_each_class(gt_boxes=customized_labels, det_boxes=predicted_labels, iou_threshold=0.5)
    iou_distances_classes = iou_dist_each_class(predicted_boxes=predicted_labels, custom_boxes=customized_labels)
    present_classes_indices = list(iou_distances_classes.keys())
    
    # If there were no matching boxes at all, consider this a high risk request
    if not present_classes_indices:
        return 1.0

    present_val_AP_classes = np.array(val_AP_classes)[present_classes_indices]
    present_iou_distance_classes = np.array([iou_distances_classes[i] for i in present_classes_indices])
    
    class_scores = weighted_geometric_mean(present_val_AP_classes, present_iou_distance_classes, validation_weight=0.4, distance_weight=0.6)
    return combine_class_scores(class_scores, weight=3)