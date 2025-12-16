import numpy as np
import matplotlib.pyplot as plt
import sap
import rasterio as rio
from .attribute_std import attribute_standard_deviation


def spectrum2d_with_custom_attrs(tree, x_attribute, y_attribute, x_count=100, y_count=100,
                                 x_log=False, y_log=False, weighted=True, normalized=True,
                                 node_mask=None, image=None):
    """
    Compute 2D pattern spectrum with support for custom attributes like standard_deviation.
    
    Parameters
    ----------
    tree : sap.Tree
        The tree used for creating the pattern spectrum
    x_attribute : str
        The name of the attribute to be used on the x-axis
    y_attribute : str
        The name of the attribute to be used on the y-axis
    x_count : int, optional
        The number of bins along the x-axis (default: 100)
    y_count : int, optional
        The number of bins along the y-axis (default: 100)
    x_log : bool, optional
        If True, the x-axis will be set to log scale (default: False)
    y_log : bool, optional
        If True, the y-axis will be set to log scale (default: False)
    weighted : bool, optional
        If True, the pattern spectrum is weighted (default: True)
    normalized : bool, optional
        If True, weights are normalized with image size (default: True)
    node_mask : ndarray, optional
        Boolean mask array of nodes to consider
    image : ndarray, optional
        Input image (required if x_attribute or y_attribute is 'standard_deviation')
        
    Returns
    -------
    s : ndarray, shape(x_count, y_count)
        The pattern spectrum
    xedges : ndarray, shape(x_count + 1,)
        The bin edges along the x-axis
    yedges : ndarray, shape(y_count + 1,)
        The bin edges along the y-axis
    x_log : bool
        Parameter x_log indicating if the x-axis is a log scale
    y_log : bool
        Parameter y_log indicating if the y-axis is a log scale
    """
    if x_attribute == 'standard_deviation':
        if image is None:
            image = tree._image
        x = attribute_standard_deviation(tree._tree, image)
    else:
        x = tree.get_attribute(x_attribute)
    
    if y_attribute == 'standard_deviation':
        if image is None:
            image = tree._image
        y = attribute_standard_deviation(tree._tree, image)
    else:
        y = tree.get_attribute(y_attribute)
    
    bins = (sap.get_bins(x, x_count, 'geo' if x_log else 'lin'),
            sap.get_bins(y, y_count, 'geo' if y_log else 'lin'))
    
    weights = _compute_node_weights(tree) if weighted else None
    weights = weights / tree._image.size if normalized and weighted else weights
    
    node_mask = np.ones_like(x, dtype=bool) if node_mask is None else node_mask
    weights = weights[node_mask] if weighted else None
    
    s, xedges, yedges = np.histogram2d(x[node_mask], y[node_mask],
                                       bins=bins, density=None,
                                       weights=weights)
    
    return s, xedges, yedges, x_log, y_log


def _compute_node_weights(tree):
    """Compute the node weights for weighted spectra."""
    dh = tree._alt - tree._alt[tree._tree.parents()]
    area = tree.get_attribute('area')
    return area * dh


def load_drone_image(filename, band=1, downsample_factor=1):
    """
    Load and normalize a drone image from file.
    
    Parameters
    ----------
    filename : str
        Path to the image file
    band : int, optional
        Band number to read (default: 1)
    downsample_factor : int, optional
        Factor by which to downsample the image (default: 1, no downsampling)
        
    Returns
    -------
    ndarray
        Normalized image array
    """
    with rio.open(filename) as src:
        if downsample_factor > 1:
            img = src.read(
                band,
                out_shape=(
                    src.height // downsample_factor,
                    src.width // downsample_factor
                ),
                resampling=rio.enums.Resampling.average
            )
        else:
            img = src.read(band)
        
        img = img.astype(np.float32)
        img = (img - img.min()) / (img.max() - img.min())
        return img


def compute_tree_attributes(tree, image, include_std=True):
    """
    Compute multiple attributes for a tree.
    
    Parameters
    ----------
    tree : sap.Tree
        Component tree (MaxTree, MinTree, etc.)
    image : ndarray
        Input image
    include_std : bool, optional
        Whether to include standard deviation attribute (default: True)
        
    Returns
    -------
    dict
        Dictionary of attribute names to values
    """
    attributes = {}
    attributes['area'] = tree.get_attribute('area')
    attributes['compactness'] = tree.get_attribute('compactness')
    attributes['moment_of_inertia'] = tree.get_attribute('moment_of_inertia')
    
    if include_std:
        attributes['standard_deviation'] = attribute_standard_deviation(
            tree._tree, image, area=attributes['area']
        )
    
    return attributes


def multi_attribute_filter(tree, thresholds, image=None):
    """
    Filter tree nodes using multiple attribute thresholds.
    
    Parameters
    ----------
    tree : sap.Tree
        Input tree
    thresholds : dict
        Dictionary of {attribute_name: (min_val, max_val)}
    image : ndarray, optional
        Input image (required if 'standard_deviation' is in thresholds)
        
    Returns
    -------
    ndarray
        Boolean mask of nodes meeting all criteria
    """
    mask = np.ones(tree.num_nodes(), dtype=bool)
    
    for attr_name, (min_val, max_val) in thresholds.items():
        if attr_name == 'standard_deviation':
            if image is None:
                image = tree._image
            vals = attribute_standard_deviation(tree._tree, image)
        else:
            vals = tree.get_attribute(attr_name)
        
        mask &= (vals >= min_val) & (vals <= max_val)
    
    return mask


def compute_2d_pattern_spectra(tree, attr_pairs, x_count=80, y_count=80, x_log=True, y_log=False, image=None):
    """
    Compute 2D pattern spectra for multiple attribute pairs.
    
    Parameters
    ----------
    tree : sap.Tree
        Component tree
    attr_pairs : list of tuples
        List of (attr1, attr2) pairs
    x_count : int, optional
        Number of bins for x-axis (default: 80)
    y_count : int, optional
        Number of bins for y-axis (default: 80)
    x_log : bool, optional
        Use logarithmic scale for x-axis (default: True)
    y_log : bool, optional
        Use logarithmic scale for y-axis (default: False)
    image : ndarray, optional
        Input image (required if any attribute is 'standard_deviation')
        
    Returns
    -------
    list
        List of spectrum results (attr1, attr2, spectrum, x_edges, y_edges, x_labels, y_labels)
        
    Notes
    -----
    Standard deviation attribute is now supported through spectrum2d_with_custom_attrs.
    """
    spectra = []
    
    for attr1, attr2 in attr_pairs:
        if attr1 == 'standard_deviation' or attr2 == 'standard_deviation':
            s, xe, ye, xl, yl = spectrum2d_with_custom_attrs(
                tree, attr1, attr2,
                x_count=x_count, y_count=y_count,
                x_log=x_log, y_log=y_log,
                image=image
            )
        else:
            s, xe, ye, xl, yl = sap.spectrum2d(
                tree, attr1, attr2, 
                x_count=x_count, y_count=y_count, 
                x_log=x_log, y_log=y_log
            )
        spectra.append((attr1, attr2, s, xe, ye, xl, yl))
    
    return spectra
    
    return spectra


def visualize_pattern_spectra(spectra, figsize=(14, 5), log_scale=True):
    """
    Visualize 2D pattern spectra.
    
    Parameters
    ----------
    spectra : list
        List of spectrum results from compute_2d_pattern_spectra
    figsize : tuple, optional
        Figure size (default: (14, 5))
    log_scale : bool, optional
        Use logarithmic scale for display (default: True)
    """
    n_spectra = len(spectra)
    fig, axes = plt.subplots(1, n_spectra, figsize=(figsize[0] * n_spectra // 2, figsize[1]))
    
    if n_spectra == 1:
        axes = [axes]
    
    for idx, (ax, (attr1, attr2, s, xe, ye, xl, yl)) in enumerate(zip(axes, spectra)):
        plt.sca(ax)
        sap.show_spectrum(s, xe, ye, xl, yl, log_scale=log_scale)
        ax.set_xlabel(attr1.replace('_', ' ').title())
        ax.set_ylabel(attr2.replace('_', ' ').title())
    
    plt.tight_layout()
    return fig


def visualize_filtering_results(original, filtered, difference=None, titles=None, figsize=(18, 6)):
    """
    Visualize original image, filtered result, and optionally the difference.
    
    Parameters
    ----------
    original : ndarray
        Original image
    filtered : ndarray
        Filtered image
    difference : ndarray, optional
        Difference image (if None, computed as original - filtered)
    titles : list, optional
        List of titles for each subplot
    figsize : tuple, optional
        Figure size (default: (18, 6))
    """
    if difference is None:
        difference = original - filtered
    
    if titles is None:
        titles = ['Original', 'Filtered Result', 'Removed Structures']
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title(titles[0])
    axes[0].axis('off')
    
    axes[1].imshow(filtered, cmap='gray')
    axes[1].set_title(titles[1])
    axes[1].axis('off')
    
    axes[2].imshow(difference, cmap='gray')
    axes[2].set_title(titles[2])
    axes[2].axis('off')
    
    plt.tight_layout()
    return fig


def process_drone_image(image, thresholds, tree_type='max', compute_spectra=True, 
                        spectrum_pairs=None):
    """
    Process a drone image with multi-attribute filtering.
    
    Parameters
    ----------
    image : ndarray or str
        Input image array or path to image file
    thresholds : dict
        Dictionary of attribute thresholds
    tree_type : str, optional
        Type of tree to build: 'max', 'min', or 'alpha' (default: 'max')
    compute_spectra : bool, optional
        Whether to compute pattern spectra (default: True)
    spectrum_pairs : list of tuples, optional
        Attribute pairs for pattern spectra
        
    Returns
    -------
    dict
        Dictionary containing tree, attributes, mask, filtered image, and optionally spectra
    """
    if isinstance(image, str):
        image = load_drone_image(image)
    
    if tree_type == 'max':
        tree = sap.MaxTree(image)
    elif tree_type == 'min':
        tree = sap.MinTree(image)
    elif tree_type == 'alpha':
        tree = sap.AlphaTree(image)
    else:
        raise ValueError(f"Unknown tree type: {tree_type}")
    
    attributes = compute_tree_attributes(tree, image, include_std='standard_deviation' in thresholds)
    
    mask = multi_attribute_filter(tree, thresholds, image=image)
    filtered = tree.reconstruct(~mask)
    
    results = {
        'tree': tree,
        'image': image,
        'attributes': attributes,
        'mask': mask,
        'filtered': filtered,
        'difference': image - filtered
    }
    
    if compute_spectra and spectrum_pairs:
        spectra = compute_2d_pattern_spectra(tree, spectrum_pairs, image=image)
        results['spectra'] = spectra
    
    return results


def batch_process_images(image_list, thresholds, **kwargs):
    """
    Process multiple images with the same thresholds.
    
    Parameters
    ----------
    image_list : list
        List of image arrays or file paths
    thresholds : dict
        Dictionary of attribute thresholds
    **kwargs
        Additional arguments passed to process_drone_image
        
    Returns
    -------
    list
        List of result dictionaries
    """
    results = []
    for img in image_list:
        result = process_drone_image(img, thresholds, **kwargs)
        results.append(result)
    return results