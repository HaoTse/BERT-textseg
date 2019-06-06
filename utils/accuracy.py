import segeval
import operator


def convert_segeval_format(result, mask):
    length = sum(mask)
    result[-1] = 1 # initial the last element is segment point
    ret = [0] + [i + 1 for i, x in enumerate(result) if x != 0]

    if len(ret) == 1:
        raise RuntimeError('Segeval format convert error')
    else:
        return tuple(map(operator.sub, ret[1:], ret[:-1]))

def acc_computer(pred, gold, window_size=-1):
    if window_size == -1:
        pk_score = segeval.pk(pred, gold)
        # windiff score cannot be applied in length smaller than 2
        windiff_score = segeval.window_diff(pred, gold) if sum(pred) > 2 else 0
    else:
        pk_score = segeval.pk(pred, gold, window_size=window_size)
        windiff_score = segeval.window_diff(pred, gold, window_size=window_size) if sum(pred) > 2 else 0
    b_score = segeval.boundary_similarity(pred, gold)

    return pk_score, windiff_score, b_score