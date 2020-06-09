import numpy as np
import scipy

from sklearn.decomposition import PCA

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

def phase_track(segments, length, normalize=False, n_components=2):
    '''
    Get phase trajectory projection of series.
    :param segments: 2Darray of shape [duration, 1]
    :param length: dimensionality of feature space.
    :param n_components: Number of components to keep
    while applying PCA to resulting trajectory.
    :return:
    - projection: projection of phase trajectory
    on the principal components.
    - basis: principal axes in feature space.
    '''
    
    if normalize:
        phase = normalize_segments(segments, length=length)
    else:
        phase = segments

    model = PCA(n_components=n_components)
    projection = model.fit_transform(phase)
    basis = model.components_
    print('Explained variation'
          ' for {} principal components: {}'.format(n_components,
                                                    model.explained_variance_ratio_))
    print('Cumulative explained variation'
          'for {} principal components: {}\n'.format(n_components,
                                                     np.sum(model.explained_variance_ratio_)))
    return projection, basis

def to_phase_trajectory(series, l):
    '''
    Get phase trajectory of series.
    Parameters:
    -series: 2Darray of shape [duration, 1]
    -l: dimensionality of feature space.
    Output:
    -phase: phase trajectory
    '''

    phase = np.zeros([series.shape[0] - l, l])

    for i in range(0, series.shape[0] - l):
        phase[i] = np.squeeze(series[i:i + l])
    return phase