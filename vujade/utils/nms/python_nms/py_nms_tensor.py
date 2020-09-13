import torch


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(_boxes, _scores, _thres=0.5, _top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        _boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        _scores: (tensor) The class predscores for the img, Shape:[num_priors].
        _thres: (float) The overlap thresh for suppressing unnecessary boxes.
        _top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = torch.Tensor(_scores.size(0)).fill_(0).long()
    if _boxes.numel() == 0:
        return keep
    x1 = _boxes[:, 0]
    y1 = _boxes[:, 1]
    x2 = _boxes[:, 2]
    y2 = _boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = _scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-_top_k:]  # indices of the top-k largest vals
    xx1 = _boxes.new()
    yy1 = _boxes.new()
    xx2 = _boxes.new()
    yy2 = _boxes.new()
    w = _boxes.new()
    h = _boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= _thres
        idx = idx[IoU.le(_thres)]
    return keep, count