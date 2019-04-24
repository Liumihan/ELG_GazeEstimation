import numpy as np


def get_peek_points(heatmap):
    """
    :param heatmap: np.ndarray, (17, 96, 160)
    :return: np.ndarray, (17, 2)
    """
    y, x = np.where(heatmap == heatmap.max())
    if len(y) > 0:
        y = y[0]
    if len(x) > 0:
        x = x[0]

    return x, y

# todo 搞好这个东西
def get_mse(pred_heatmaps, target_points):
    pred_points = np.zeros(shape=target_points.shape)
    for i, heatmap in enumerate(pred_heatmaps):
        x, y = get_peek_points(heatmap)
        pred_points[i, :]

