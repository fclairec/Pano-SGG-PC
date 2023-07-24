import numpy as np
from plyfile import PlyData, PlyElement
from collections import namedtuple

class PlyDataReader:
    def __init__(self):
        self._pos = None
        self._label = None

    @property
    def pos(self):
        return self._pos

    @property
    def label(self):
        return self._label

    def read_ply(self, path):
        with open(path, 'rb') as f:
            ply_data = PlyData.read(f)

        vertex_data = ply_data['vertex']
        pos = np.stack((vertex_data['x'], vertex_data['y'], vertex_data['z']), axis=-1)
        if 'scalar_Scalar_field' in [i.name for i in vertex_data.properties]:
            label = vertex_data['scalar_Scalar_field']
        else:
            label = None

        Data = namedtuple('Data', ['pos', 'label'])
        data = Data(pos=pos, label=label)

        if self._pos is not None and self._label is not None:
            self._pos = np.concatenate((self._pos, data.pos), axis=0)
            self._label = np.concatenate((self._label, data.label), axis=0)
        else:
            self._pos = data.pos
            self._label = data.label

    def pcd2ply(self, points, labels):
        if self._pos is not None and self._label is not None:
            self._pos = np.concatenate((self._pos, points), axis=0)
            self._label = np.concatenate((self._label, labels), axis=0)
        else:
            self._pos = points
            self._label = labels

    def write_ply(self, path):
        if self._pos is not None and self._label is not None:
            vertex = np.array(list(zip(self._pos[:, 0], self._pos[:, 1], self._pos[:, 2], self._label)),
                              dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('scalar_Scalar_field', 'f4')])

            el = PlyData([PlyElement.describe(vertex, 'vertex')])
            el.write(path)