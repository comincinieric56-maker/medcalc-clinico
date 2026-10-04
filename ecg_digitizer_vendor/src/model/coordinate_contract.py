"""Keep extractor samples in the aligned feature map's pixel coordinates."""


def restore_aligned_lines(lines, canvas_width, crop_bounds):
    if lines.ndim != 2 or canvas_width <= 0:
        raise ValueError('INVALID_ALIGNED_LINE_SHAPE')
    if crop_bounds is None:
        if lines.shape[1] != canvas_width:
            raise ValueError('EXTRACTOR_COORDINATES_UNAVAILABLE')
        return lines, None
    left, right = crop_bounds
    if not 0 <= left <= right < canvas_width:
        raise ValueError('INVALID_EXTRACTOR_CROP_BOUNDS')
    if lines.shape[1] != right-left+1:
        raise ValueError('EXTRACTOR_CROP_WIDTH_MISMATCH')
    restored = lines.new_full((lines.shape[0], canvas_width), float('nan'))
    restored[:, left:right+1] = lines
    return restored, [left, right]
