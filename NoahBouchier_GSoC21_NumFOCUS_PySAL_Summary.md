# Noah Bouchier - Google Summer of Code summary

Hey there :wave: ,

To NumFOCUS, Google, or anyone else interested in reading a summary of my experience with Google Summer of Code (GSoC), welcome!

# Key links

- ***[Pull Request to add function and documentation to PySAL segregation package](https://github.com/pysal/segregation/pull/185)***
- ***[Completed function](https://github.com/noahbouchier/segregation/blob/master/segregation/spatial/kl_divergence_profile.py)***
- ***[Completed documentation](https://github.com/noahbouchier/segregation/blob/master/notebooks/kl_divergence_profile_walkthrough.ipynb)


- [GSoC project page](https://summerofcode.withgoogle.com/projects/#6509416444067840)
- [GitHub repository for project](https://github.com/noahbouchier/GSoC-PySAL-21)
- [Personal blog of project](https://noahbouchier.github.io/blog/gsoc/)

# Personal reflection and gratitude

First of all, let me start by saying I have had a *fantastic* time being a part of GSoC. As a programmer, I have boosted my skills in a coding language I had limited experience with, and have learnt skills that I will be able to transfer to many different contexts. And as a person, I have benefitted a great amount as a result of working and contributing to a team of highly-skilled experts, keeping myself accountable to tracking the progress of this project and working with international scheduling considerations to ensure that meetings are organised and communication is clear and concise.

I am so thankful to Levi and Jeff, for their dedicated mentoring, and to the wider PySAL network for the interest and support given during community meetings. Thank you to Google and NumFOCUS, for making this opportunity possible. It has been a fantastic way to spend my summer, honing my programming skills to become a more intelligent, effective contributor to the open source community.

# Project reflection and achievements

## KL Divergence Profile function

The goal of this project was to create a function that implemented the methodology presented by Madalina Olteanu, Julien Randon-Furling, and William A. V. Clark in their 2019 article ["Segregation through the multiscalar lens"](https://doi.org/10.1073/pnas.1900192116). This methodology calculated the [Kullback-Leiber (KL)](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) divergence level between a local area and that of the total population, with respect to different groups. The main motivation for this statistic is to better understand the multiscalar dynamics of ethnic segregation within urban areas.

Following the completion of the Google Summer of Code, this function has been created, and is now in the process of being implemented into the PySAL library. The release of this code, at the end of GSoC, can be seen below:

<details>
  <summary>**KL Divergence Profile function - end of GSoC '21 code release**</summary>

``` python
import numpy as np
import geopandas as gpd
import pandas as pd

from scipy.spatial.distance import pdist, squareform
from scipy.special import rel_entr as relative_entropy


def kl_divergence_profile(populations, coordinates = None, metric = 'euclidean'):
    """
    A segregation metric, using Kullback-Leiber (KL) divergence to quantify the
    difference in the population characteristics between (1) an area and (2) the total population.

    This function utilises the methodology proposed in
    Olteanu et al. (2019): 'Segregation through the multiscalar lens'. Which can be
    found here: https://doi.org/10.1073/pnas.1900192116

    Arguments
    ----------
    populations : GeoPandas GeoDataFrame object
                  NumPy Array object
                  Population information of raw group numbers (not percentages) to be
                  included in the analysis.
    coordinates : GeoPandas GeoSeries object
                  NumPy Array object
                  Spatial information relating to the areas to be included in the analysis.
    metric : Acceptable inputs to `scipy.spatial.distance.pdist` - including:
             ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’,
             ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’,
             ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’,
             ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’.
             Distance metric for calculating pairwise distances,
             using `scipy.spatial.distance.pdist` - 'euclidean' by default.

    Returns
    ----------
    Returns a concatenated object of Pandas dataframes. Each dataframe contains a
    set of divergence levels between an area and the total population. These areas
    become consecutively larger, starting from a single location and aggregating
    outward from this location, until the area represents the total population.
    Thus, together the divergence levels within a dataframe represent a profile
    of divergence from an area. The concatenated object is the collection of these
    divergence profiles for every areas within the total population.

    Example
    ----------
    from libpysal.examples import get_path
    from libpysal.examples import load_example
    cincin = load_example('Cincinnati')
    cincin.get_file_list()
    cincin_df = gpd.read_file(cincin.get_path('cincinnati.shp'))
    cincin_ethnicity = cincin_df[["WHITE", "BLACK", "AMINDIAN", "ASIAN", "HAWAIIAN", "OTHER_RACE", "geometry"]]
    cincin_ethnicity.head()
    kl_divergence_profile(cincin_ethnicity)
    """
    # Store the observation index to return with the results
    if hasattr(populations, 'index'):
        indices = populations.index
    else:
        indices = np.arange(len(populations))

    # Check for geometry present in populations argument
    if hasattr(populations, 'geometry'):
        if coordinates is None:
            coordinates = populations.geometry
        populations = populations.drop(populations.geometry.name, axis = 1).values
    populations = np.asarray(populations)

    #  Creating consistent coordinates - GeoSeries input
    if hasattr(coordinates,'geometry'):
        centroids = coordinates.geometry.centroid
        coordinates = np.column_stack((centroids.x, centroids.y))
    #  Creating consistent coordinates - Array input
    else:
        assert len(coordinates) == len(populations), "Length of coordinates input needs to be of the same length as populations input"

    # Creating distance matrix using defined metric (default euclidean distance)
    dist_matrix = squareform(pdist(coordinates, metric = metric))

    # Preparing list for results
    results = []

    # Loop to calculate KL divergence
    for (i, distances) in enumerate(dist_matrix):

        # Creating the q and r objects
        sorted_indices = np.argsort(distances)
        cumul_pop_by_group = np.cumsum(populations[sorted_indices], axis = 0)
        obs_cumul_pop = np.sum(cumul_pop_by_group, axis = 1)[:, np.newaxis]
        q_cumul_proportions = cumul_pop_by_group / obs_cumul_pop
        total_pop_by_group = np.sum(populations, axis = 0, keepdims = True)
        total_pop = np.sum(populations)
        r_total_proportions = total_pop_by_group / total_pop

        # Input q and r objects into relative entropy (KL divergence) function
        kl_divergence = relative_entropy(q_cumul_proportions,
                                         r_total_proportions).sum(axis = 1)

        # Creating an output dataframe
        output = pd.DataFrame().from_dict(dict(
            observation = indices[i],
            distance = distances[sorted_indices],
            divergence = kl_divergence,
            population_covered = obs_cumul_pop.sum(axis=1)
        ))

        # Append (bring together) all outputs into results list
        results.append(output)

    return(pd.concat(results))


```
</details>

I fully intend to continue working with the PySAL network beyond the end of this period, to ensure the smooth implementation of this function into their toolset and assist with any alterations that are required to make this possible.

## Supporting documentation

The project also aimed to provide documentation that assisted the use of this function, in the hope of diversifying the potential users of this function. To this end, an easy-to-follow notebook has been created - using an example dataset - to cover the running of this function, suggested uses for the output, and step-by-step guide through the inner workings of this function.

This workbook has taken many iterations throughout the project, which can be seen within the *[explanatory and interactive workbooks](https://github.com/noahbouchier/GSoC-PySAL-21/tree/master/Explanatory%20and%20Interactive%20Workbooks)* within the GitHub repository for this project.

Overall, I look forward to seeing this function be implemented into PySAL, and to provide any additional support in maintaining its improvement once released for wider use. I hope to use my experiences during GSoC as a stimulus to continue my involvement in PySAL and wider spatial analysis programming community. I hope to track this via my website, so please do check up on me here: [https://noahbouchier.github.io/](https://noahbouchier.github.io/).

All the best - go well,

Noah
