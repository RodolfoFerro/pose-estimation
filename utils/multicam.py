from psutil import Process


def process_viewer(viewer, pid):
    """Processor for multicam view.

    Runs the viewer itself as a subprocess.

    Parameters
    ----------
    viewer : utils.viewer.Viewer
        The Viewer object to be run. It should be previously initialized.
    pid : int
        Contains an identifier of the process used to assign CPUs.
    """


    p = Process()
    p.cpu_affinity([pid + 1])

    viewer.run()