import numpy as np  
import numba

@numba.jit(nopython=True)
def iou_jit(boxes, query_boxes, eps=0.0):
    """calculate box iou. note that jit version runs 2x faster than cython in 
    my machine!
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        box_area = ((query_boxes[k, 2] - query_boxes[k, 0] + eps) *
                    (query_boxes[k, 3] - query_boxes[k, 1] + eps))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + eps)
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + eps)
                if ih > 0:
                    ua = (
                        (boxes[n, 2] - boxes[n, 0] + eps) *
                        (boxes[n, 3] - boxes[n, 1] + eps) + box_area - iw * ih)
                    overlaps[n, k] = iw * ih / ua

    return overlaps

def bresenham(x0, y0, x1, y1):
    """Yield integer coordinates on the line from (x0, y0) to (x1, y1) in image.
    Input coordinates should be integers.
    The result will contain both the start and the end point.
    """
    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx >= dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2*dy - dx
    y = 0

    for x in range(dx + 1):
        yield x0 + x*xx + y*yx, y0 + x*xy + y*yy
        if D >= 0:
            y += 1
            D -= 2*dx
        D += 2*dy

def calculate_ellipse_pixels(center, axes, angle):
    
    Cos = np.cos(np.deg2rad(angle))
    Sin = np.sin(np.deg2rad(angle))

    coords = []

    for ang in range(0,360):

        u = np.cos(np.deg2rad(ang))*axes[0]
        v = np.sin(np.deg2rad(ang))*axes[1]

        x = u*Cos - v*Sin + center[0]
        y = u*Sin - v*Cos + center[1]
        coords.append([x,y])

    return np.round(np.array(coords),0).astype(np.int32)

def calculate_ellipse_pixel(center, axes, angle, ang):
    
    Cos = np.cos(np.deg2rad(angle))
    Sin = np.sin(np.deg2rad(angle))

    coords = []

    u = np.cos(np.deg2rad(ang))*axes[0]
    v = np.sin(np.deg2rad(ang))*axes[1]

    x = u*Sin - v*Cos + center[0]
    y = u*Cos - v*Sin + center[1]
    coords.append([x,y])

    return np.round(np.array(coords),0).astype(np.int32)

def ellipse_extremas(center_coordinates,axesLength,angle):

    extremas = np.zeros(shape=(4,))

    Sin = np.sin(np.deg2rad(angle))
    Cos = np.cos(np.deg2rad(angle))

    tx = np.arctan(-(axesLength[1]*Sin)/(axesLength[0]*Cos))
    ty = np.arctan(-(axesLength[1]*Cos)/(axesLength[0]*Sin))

    x1 = axesLength[0]*Cos*np.cos(tx) - axesLength[1]*Sin*np.sin(ty) + center_coordinates[0]
    y1 = axesLength[0]*Sin*np.cos(tx) - axesLength[1]*Cos*np.sin(ty) + center_coordinates[1]
    x2 = axesLength[0]*Cos*np.cos(tx + np.pi) - axesLength[1]*Sin*np.sin(ty + np.pi) + center_coordinates[0]
    y2 = axesLength[0]*Sin*np.cos(tx + np.pi) - axesLength[1]*Cos*np.sin(ty + np.pi) + center_coordinates[1]

    if x1 < x2:
        extremas[0] = np.round(x1,0).astype(np.int32)
        extremas[2] = np.round(x2,0).astype(np.int32)
    else:
        extremas[0] = np.round(x2,0).astype(np.int32)
        extremas[2] = np.round(x1,0).astype(np.int32)

    if y1 < y2:
        extremas[1] = np.round(y1,0).astype(np.int32)
        extremas[3] = np.round(y2,0).astype(np.int32)
    else:
        extremas[1] = np.round(y2,0).astype(np.int32)
        extremas[3] = np.round(y1,0).astype(np.int32)

def circle_center_limits(radius, width, height):

    assert width//2 > radius, 'Circle does not fit in image'
    assert height//2 > radius, 'Circle does not fit in image'
    
    return (radius, width-radius), (radius, height-radius)
    
def ellipse_center_limits(axesLength, angle, width, height):

    assert width//2 > axesLength[0], 'Ellipse does not fit in image'
    assert height//2 > axesLength[1], 'Ellipse does not fit in image'

    if angle == 0:

        return (axesLength[0], height - axesLength[0]), (axesLength[1], width - axesLength[1])

    else:

        Sin = np.sin(np.deg2rad(angle))
        Cos = np.cos(np.deg2rad(angle))

        tx = np.arctan(-(axesLength[1]*Sin)/(axesLength[0]*Cos))
        ty = np.arctan(-(axesLength[1]*Cos)/(axesLength[0]*Sin))

        x1 = axesLength[0]*Cos*np.cos(tx) - axesLength[1]*Sin*np.sin(ty)
        y1 = axesLength[0]*Sin*np.cos(tx) - axesLength[1]*Cos*np.sin(ty)
        x2 = axesLength[0]*Cos*np.cos(tx + np.pi) - axesLength[1]*Sin*np.sin(ty + np.pi)
        y2 = axesLength[0]*Sin*np.cos(tx + np.pi) - axesLength[1]*Cos*np.sin(ty + np.pi)

        x_low = np.round(np.abs(np.min([x1,x2])))
        x_up = image.shape[1]- np.round(np.max([x1,x2]))

        y_low = np.round(np.abs(np.min([y1,y2])))
        y_up = image.shape[0]- np.round(np.max([y1,y2]))

        return (int(x_low), int(x_up)), (int(y_low), int(y_up))