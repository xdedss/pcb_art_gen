
import cv2
import watchdog
import numpy as np
import yaml

import os, argparse, re

MIL = 0.001 * 0.0254
color_keys = ['bare', 'mask', 'mask_copper', 'copper', 'overlay']

def perfect_circular_element(r):
    size = r * 2 + 1
    ys, xs = np.mgrid[:size, :size]
    ys -= r
    xs -= r
    distance = np.linalg.norm(np.stack([xs, ys]), axis=0)
    return (distance < (r + 0.5)).astype(np.uint8)

def dilate_circular(img, px):
    ''' dilate using perfect_circular_element '''
    max_r = 7
    if (px > max_r):
        iterations = px // max_r
        img = cv2.dilate(img, perfect_circular_element(max_r), iterations=iterations)
    img = cv2.dilate(img, perfect_circular_element(px % max_r))
    return img

def open_mask(bool_mask, kernel, iterations=1):
    ''' Morph open on bool img '''
    return cv2.morphologyEx(bool_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations) > 0

def close_mask(bool_mask, kernel, iterations=1):
    ''' Morph open on bool img '''
    return cv2.morphologyEx(bool_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations) > 0

def filter_mask_by_size(bool_mask, f_awh):
    ''' f_awh: lambda area, width, height: True/False '''
    res = bool_mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(res, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        area = cv2.contourArea(contour)
        (x, y), (w, h), angle = cv2.minAreaRect(contour)
        if (not f_awh(area, w, h)):
            cv2.fillPoly(res, [contour], color=0)
    return res > 0

def find_content_size_px(img):
    ''' find [width, height] of >0 pixels in img '''
    if (np.sum(img > 0) == 0):
        return 0, 0
    ys, xs = np.where(img > 0)
    return (1 + xs.max() - xs.min()), (1 + ys.max() - ys.min())


def parse_physical_float(s, default_unit=1.0):
    re_float = '(^[1-9]\d*\.\d+|^0\.\d+|^[1-9]\d*|^0)'
    no_unit_res = re.findall(f'{re_float}$', s)
    if (len(no_unit_res) > 0):
        return float(no_unit_res[0]) * default_unit
    mm_res = re.findall(f'{re_float}mm$', s)
    if (len(mm_res) > 0):
        return float(mm_res[0]) * 0.001
    m_res = re.findall(f'{re_float}m$', s)
    if (len(m_res) > 0):
        return float(m_res[0]) * 1
    mil_res = re.findall(f'{re_float}mil$', s)
    if (len(mil_res) > 0):
        return float(mil_res[0]) * MIL
    raise Exception(f'Invalid format: {s}')


def get_masks(img, color_config, overlay_clearance_px=2, min_line_width_px=3, min_gap_width_px=3, min_copper_size_px=5):
    color_ids = {k: i for i, k in enumerate(color_keys)}
    # 1. segmentation
    color_distances = []
    for k in color_keys:
        color = np.array(color_config[k])
        distance = np.linalg.norm(img - color[None, None, :], axis=2)
        color_distances.append(distance)
    seg_map = np.argmin(np.stack(color_distances, axis=0), axis=0)
    # 2. decomp
    overlay_clearance = dilate_circular((seg_map == color_ids['overlay']).astype(np.uint8), overlay_clearance_px) > 0
    solder_mask = ((seg_map == color_ids['bare']) | (seg_map == color_ids['copper'])) & np.bitwise_not(overlay_clearance)
    copper_mask = (seg_map == color_ids['mask_copper']) | (seg_map == color_ids['copper'])
    overlay_mask = (seg_map == color_ids['overlay'])
    # open
    open_kernel = perfect_circular_element(min_line_width_px // 2)
    close_kernel = perfect_circular_element(min_gap_width_px // 2)
    solder_mask = open_mask(solder_mask, open_kernel)
    copper_mask = open_mask(copper_mask, open_kernel)
    overlay_mask = open_mask(overlay_mask, open_kernel)
    solder_mask = close_mask(solder_mask, close_kernel)
    copper_mask = close_mask(copper_mask, close_kernel)
    overlay_mask = close_mask(overlay_mask, close_kernel)
    # remove dead copper
    copper_mask = filter_mask_by_size(copper_mask, lambda a, w, h: min(w, h) > (min_copper_size_px))

    return solder_mask, copper_mask, overlay_mask

def get_preview(solder_mask, copper_mask, overlay_mask, color_config):
    preview = np.zeros((*solder_mask.shape[:2], 3), dtype=np.uint8)
    # bg
    preview[:, :] = color_config['bare']
    preview[copper_mask] = color_config['copper']
    preview[np.bitwise_not(solder_mask) & copper_mask] = color_config['mask_copper']
    preview[np.bitwise_not(solder_mask) & np.bitwise_not(copper_mask)] = color_config['mask']
    preview[np.bitwise_not(solder_mask) & overlay_mask] = color_config['overlay']

    return preview

def mask_to_contours(img, approx_eps_px=1.0):
    h, w = img.shape[:2]
    contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
    res = []
    for contour in contours:
        contour = cv2.approxPolyDP(contour, approx_eps_px, closed=True)
        res.append(contour)
    return res

def contours_to_mask(contours, w, h):
    res = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(res, contours, -1, color=1, thickness=-1)
    return res > 0

def contours_to_svg(filename, contours, w, h, scale=1.0, fill="black"):
    with open(filename, 'w') as f:
        f.write(f'<svg width="{w * scale}px" height="{h * scale}px" viewBox="0 0 {w * scale} {h * scale}" xmlns="http://www.w3.org/2000/svg" version="1.1">\n')
        f.write(f'<path d="\n')
        for contour in contours:
            '''
            M 100 100 L 200 100 L 200 200 L 100 200 z
            M 120 120 L 190 110 L 190 190 L 110 190 z
            '''
            for i, ((x, y), ) in enumerate(contour):
                f.write(f'{"M" if i == 0 else "L"} {x * scale} {y * scale} ')
            f.write('z \n')
        f.write(f'" fill="{fill}" fill-rule="evenodd"/>\n')
        f.write(f'</svg>\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_file')
    parser.add_argument('color_config') # in BGR
    parser.add_argument('-o', '--output', type=str, default=None, help='output directory')
    # parser.add_argument('-i', '--interactive', action='store_true')
    parser.add_argument('-w', '--width', type=str, default='50mm', help='expected physical width of the full image')
    parser.add_argument('-r', '--resolution', type=str, default='1mil', help='physical resolution of output png')
    parser.add_argument('--overlay-clearance', type=str, default='2mil', help='minimum solder mask around overlay')
    parser.add_argument('--min-line-width', type=str, default='12mil', help='size of morph open kernel')
    parser.add_argument('--min-gap-width', type=str, default='12mil', help='size of morph close kernel')
    parser.add_argument('--min-copper-size', type=str, default='1mm', help='dead copper whos min(w, h) < this will be removed')
    parser.add_argument('--svg-approx-eps', type=str, default='2mil', help='precision of output svg')
    args = parser.parse_args()

    # check file
    assert(os.path.isfile(args.image_file))
    assert(os.path.isfile(args.color_config))

    with open(args.color_config, 'r') as f:
        color_config = yaml.load(f, yaml.Loader)
    assert np.all([k in color_config for k in color_keys])

    img = cv2.imread(args.image_file)
    assert img is not None

    # check param
    physical_width = parse_physical_float(args.width)
    overlay_clearance = parse_physical_float(args.overlay_clearance)
    min_line_width = parse_physical_float(args.min_line_width)
    min_gap_width = parse_physical_float(args.min_gap_width)
    target_resolution = parse_physical_float(args.resolution)
    svg_approx_eps = parse_physical_float(args.svg_approx_eps)
    min_copper_size = parse_physical_float(args.min_copper_size)

    # output dir
    output_dir = os.path.splitext(os.path.basename(args.image_file))[0] + '_output'
    print('output to: ', output_dir)
    os.makedirs(output_dir, exist_ok=True)
    o = lambda f: os.path.join(output_dir, f)

    # preprocessing
    img = cv2.medianBlur(img, ksize=3)

    # rescale
    current_resolution = physical_width / img.shape[1]
    scale_factor = current_resolution / target_resolution
    img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    h_px, w_px = img.shape[:2]

    # 1. pixelated mask
    solder_mask, copper_mask, overlay_mask = get_masks(img, color_config, 
        overlay_clearance_px=int(np.ceil(overlay_clearance / target_resolution)),
        min_line_width_px=int(np.ceil(min_line_width / target_resolution)),
        min_gap_width_px=int(np.ceil(min_gap_width / target_resolution)),
        min_copper_size_px=int(np.ceil(min_copper_size / target_resolution)) )
    preview = get_preview(solder_mask, copper_mask, overlay_mask, color_config)

    print(f'solder_mask mask width: {find_content_size_px(solder_mask)[0] * target_resolution * 1000} mm')
    print(f'copper_mask mask width: {find_content_size_px(copper_mask)[0] * target_resolution * 1000} mm')
    print(f'overlay_mask mask: {find_content_size_px(overlay_mask)[0] * target_resolution * 1000} mm')

    cv2.imwrite(o('solder_mask.png'), 255 * (1 - solder_mask.astype(np.uint8)))
    cv2.imwrite(o('copper_mask.png'), 255 * (1 - copper_mask.astype(np.uint8)))
    cv2.imwrite(o('overlay_mask.png'), 255 * (1 - overlay_mask.astype(np.uint8)))
    cv2.imwrite(o('preview.png'), preview)

    # 2. contour -> svg
    approx_eps_px = svg_approx_eps / target_resolution
    solder_mask_contours = mask_to_contours(solder_mask, approx_eps_px=approx_eps_px)
    copper_mask_contours = mask_to_contours(copper_mask, approx_eps_px=approx_eps_px)
    overlay_mask_contours = mask_to_contours(overlay_mask, approx_eps_px=approx_eps_px)
    preview_contours = get_preview(
        contours_to_mask(solder_mask_contours, w_px, h_px),
        contours_to_mask(copper_mask_contours, w_px, h_px),
        contours_to_mask(overlay_mask_contours, w_px, h_px),
        color_config
    )

    scale_to_1mil = target_resolution / (1 * MIL)
    contours_to_svg(o('solder_mask.svg'), solder_mask_contours, w_px, h_px, scale=scale_to_1mil)
    contours_to_svg(o('copper_mask.svg'), copper_mask_contours, w_px, h_px, scale=scale_to_1mil)
    contours_to_svg(o('overlay_mask.svg'), overlay_mask_contours, w_px, h_px, scale=scale_to_1mil)
    cv2.imwrite(o('preview_svg.png'), preview_contours)
    


if __name__ == '__main__':
    main()


