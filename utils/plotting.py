from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from pytorch3d.ops import sample_points_from_meshes
mpl.rcParams['savefig.dpi'] = 80
mpl.rcParams['figure.dpi'] = 80

def plot_pointcloud(mesh, title=""):
    """
    This function plot a mesh as pointcloud for neptune.ai it can be usefull
    Args:
        mesh: a mesh element
        title: The title of the image

    Returns: the figure object

    """
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    return fig