import numpy as np
import scipy

def shrink_segment(segment, output_length):
    segment_indices = np.arange(segment.size)
    interpolated_f = scipy.interpolate.interp1d(segment_indices,
                                                segment,
                                                kind='cubic')
    new_indices = np.linspace(0, segment_indices[-1], output_length)
    return interpolated_f(new_indices)

def normalize_segments(segments, length=None):
    segments = list(segments)
    length = length if length else min(segment.size for segment in segments)
    return [shrink_segment(segment, length) for segment in segments]
