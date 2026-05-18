import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.cluster.hierarchy import dendrogram, linkage
import math
import numpy as np
from PIL import Image
import urllib

def getImage(path, zoom=1):
  '''
  Auxiliary function to get image from path
  and return it as OffsetImage object.
  '''
  with urllib.request.urlopen(path) as url_file:
    image = Image.open(url_file)
  
  return OffsetImage(image, zoom=zoom)

def plot_centroids (
  dataframe: pd.DataFrame, 
  x: str,
  y: str,
  xlab: str,
  ylab: str,
  centroids: list
)-> None:
    '''
    Function to plot centroids on top a scatter plot of the
    data frame with variables x and y.
    
    Arguments: 
    - dataframe(pd.DataFrame): DataFrame with variables to plot.
    - x(string): String with the name of the variable to plot on the x-axis.
    - y(string): String with the name of the variable to plot on the y-axis.
    - xlab(string): String of x-axis label.
    - ylab(string): String of y-axis label.
    - centroids(list): A list of lists with the centroids in the
    format of [x, y].
    
    Returns:
    - None, although a plot is produced.
    '''
    
    # Plot scatter of Number of Transactions vs. Transaction
    # Amount
    plt.scatter(
        dataframe[x],
        dataframe[y],
        s=100
    )
    
    # Plot each centroid with an orange color.
    for centroid in centroids:
        plt.scatter(
            centroid[0],
            centroid[1],
            c='orange',
            s=200
        )  

    # Matplotlib extras
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title('{} vs. {}'.format(x, y))
    # plt.show()
  

def plot_centroids_customers(
  dataframe: pd.DataFrame, 
  x: str,
  y: str,
  xlab: str,
  ylab: str,
  centroids: list,
  image_url: str,
  image_pos_x: int,
  image_pos_y: int
)-> None:
    '''
    Plots centroids and scatter plot with
    specific emoji on position image_pos_x and
    image_pos_y.
    Arguments: 
    - dataframe(pd.DataFrame): DataFrame with variables to plot.
    - x(string): String with the name of the variable to plot on the x-axis.
    - y(string): String with the name of the variable to plot on the y-axis.
    - xlab(string): String of x-axis label.
    - ylab(string): String of y-axis label.
    - centroids(list): A list of lists with the centroids in the
    format of [x, y].
    - image_url(string): url with the emoji to plot.
    - image_pos_x(int): integer value for position x of the emoji.
    - image_pos_y(int): integer value for position y of the emoji.
    
    Returns:
    - ax(matplotlib.pyplot plot): Plot with centroids, scatter and emoji.
    '''
    fig, ax = plt.subplots()
    # Plot Scatter with Data Points
    plot_centroids(
      dataframe,
      x,
      y,
      xlab,
      ylab, 
      [[30, 110], [80, 20]]
    )

    # Plot Image
    ab = AnnotationBbox(
      getImage(image_url, zoom=0.3), 
      (image_pos_x, image_pos_y), 
      frameon=False)
    ax.add_artist(ab)

    return ax
 
def plot_clusters(
  data: pd.DataFrame,
  cluster_column: str,
  norm_limit = None
) -> None:
    '''
    Plots scatter with color according to cluster.
    Arguments:
    - data(pd.DataFrame): dataframe with the data to pot.
    - cluster_columns(str): the cluster column to plot.
    - norm_limit(str): If we want to normalize axis to the same
    length.
    Returns:
    - None, although a plot is shown.
    '''
    plt.scatter(
        data.age,
        data.annual_income,
        s=100,
        c=data[cluster_column]
        )
    
    if norm_limit:
      # Throwing same limits on the plot:
      plt.xlim(15, 140)
      plt.ylim(15, 140)



def plot_customers_hierarchical(data, image_data):
  '''
  Plots hierarchical customers data with a list of emojis.
  Emojis are passed to the plot using a dictionary.
  Arguments:
  - data(pd.DataFrame): The dataframe with the scatters to plot.
  - image_data(dict): The dictionary with an url and coordinates to
  plot the emoji.
  Returns:
  - ax(matplotlib Figure): Plot with emojis on top of scatter.
  '''
  fig, ax = plt.subplots()
  ax.scatter(data.age, data.annual_income, s=200)
  plt.xlabel('Customer Age')
  plt.ylabel('Annual Income')
  plt.xlabel('Customer Age')
  plt.ylabel('Annual Income')

  for url, coordinates in image_data.items():
    # Plot Image
    ab = AnnotationBbox(
      getImage(url, zoom=0.2), 
      (coordinates[0], coordinates[1]), 
      frameon=False)
    ax.add_artist(ab)

  return ax

def plot_hierarchical_cluster(data, image_data, cluster_list):
  '''
  Plots clusters with radius on scatter plot.
  Builds upon plot build by plot_customers_hierarchical.
  Arguments:
  - data(pd.DataFrame): The dataframe with the scatters to plot.
  - image_data(dict): Dictionary with emoji urls and coordinates.
  - cluster_list(list): List of lists with centroids.
  Returns:
  - None, but a plot is shown
  '''
  ax = plot_customers_hierarchical(data, image_data)
  plt.xlim(15, 140)
  plt.ylim(15, 140)
  ax.scatter(data.age, 
             data.annual_income, 
             s=200, 
             c='white', 
             edgecolors='white')

  for circles in cluster_list:
    el_1 = plt.Circle(
        (circles[0], circles[1]), 
        radius=circles[2], 
        color='black', 
        fill=False)
    ax.add_patch(el_1)
  plt.show()


def pairwise_euclidean_distances(list1, list2):
  '''
  Returns pairwise euclidean distance.
  '''
  distance = 0
  for i in range(len(list1)):
    distance += (list1[i] - list2[i]) ** 2
  return math.sqrt(distance)

def plot_dendrogram(model, **kwargs):
    '''
    Create linkage matrix and then plot the dendrogram
    Arguments: 
    - model(HierarchicalClustering Model): hierarchical clustering model.
    - **kwargs
    Returns:
    None, but dendrogram plot is produced.
    '''
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)