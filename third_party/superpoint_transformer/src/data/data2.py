import copy
import h5py
import torch
import warnings
import numpy as np
from time import time
from typing import (
    List,
    Tuple,
    Optional,
    Union,
    Any,
    Dict)
from torch_geometric.data import Data as PyGData
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.nn.pool.consecutive import consecutive_cluster

import src
from src.data.csr import CSRData
from src.data.cluster import Cluster
from src.data.instance import InstanceData
from src.metrics import (
    SemanticMetricResults,
    PanopticMetricResults,
    InstanceMetricResults)
from src.utils import (
    tensor_idx,
    is_arange,
    is_dense,
    has_duplicates,
    isolated_nodes,
    knn_2,
    save_tensor,
    load_tensor,
    save_tensor_dict,
    load_tensor_dict,
    save_dense_to_csr,
    load_csr_to_dense,
    to_trimmed,
    to_float_rgb,
    to_byte_rgb,
    from_flat_tensor,
    human_readable_memory)


__all__ = ['Data']


class Data(PyGData):
    """Inherit from torch_geometric.Data with extensions tailored to our
    specific needs.
    """

    _NOT_INDEXABLE = [
        '_csr_', '_cluster_', '_instance_data_', 'edge_index', 'edge_attr',
        '_slice_dict', '_inc_dict', '_num_graphs']


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if src.is_debug_enabled():
            self.debug()

    @property
    def pos(self):
        return self['pos'] if 'pos' in self._store else None

    @property
    def rgb(self):
        return self['rgb'] if 'rgb' in self._store else None

    @property
    def obj(self) -> InstanceData:
        """InstanceData object indicating the instance indices for each
        node/point/superpoint in the Data.
        """
        return self['obj'] if 'obj' in self._store else None

    @property
    def semantic_pred(self):
        return self['semantic_pred'] if 'semantic_pred' in self._store else None

    @property
    def neighbor_index(self):
        return self['neighbor_index'] if 'neighbor_index' in self._store \
            else None

    @property
    def sub(self) -> Cluster:
        """Cluster object indicating subpoint indices for each point."""
        return self['sub'] if 'sub' in self._store else None

    @property
    def super_index(self):
        """Index of the superpoint each point belongs to."""
        return self['super_index'] if 'super_index' in self._store else None

    @property
    def v_edge_attr(self):
        """Vertical edge features."""
        return self['v_edge_attr'] if 'v_edge_attr' in self._store else None

    def norm_index(self, mode: str = 'graph') -> torch.Tensor:
        """Index to be used for LayerNorm.

        :param mode: str
            Normalization mode. 'graph' will normalize per graph (i.e.
            per cloud, i.e. per batch). 'node' will normalize per node
            (i.e. per point). 'segment' will normalize per segment
            (i.e.  per cluster)
        """
        if getattr(self, 'batch', None) is not None:
            batch = self.batch
        else:
            batch = torch.zeros(
                self.num_nodes, device=self.device, dtype=torch.long)
        if self.super_index is not None:
            super_index = self.super_index
        else:
            super_index = torch.zeros(
                self.num_nodes, device=self.device, dtype=torch.long)
        if mode == 'graph':
            return batch
        elif mode == 'node':
            return torch.arange(self.num_nodes, device=self.device)
        elif mode == 'segment':
            num_batches = batch.max() + 1
            return super_index * num_batches + batch
        else:
            raise NotImplementedError(f"Unknown mode='{mode}'")

    @property
    def is_super(self):
        """Whether the points are superpoints for a denser sub-graph."""
        return self.sub is not None

    @property
    def is_sub(self):
        """Whether the points belong to a coarser super-graph."""
        return self.super_index is not None

    @property
    def has_neighbors(self):
        """Whether the points have neighbors."""
        return self.neighbor_index is not None and self.neighbor_index.shape[1] > 0

    @property
    def has_edges(self):
        """Whether the points have edges."""
        return self.edge_index is not None and self.edge_index.shape[1] > 0

    @property
    def has_edge_attr(self):
        """Whether the edges have features in `edge_attr`."""
        return self.edge_attr is not None and self.edge_attr.shape[0] > 0

    @property
    def edge_keys(self) -> List[str]:
        """All keys starting with `edge_`, apart from `edge_index` and
        `edge_attr`.
        """
        return [
            k for k in self.keys
            if k.startswith('edge_') and k not in ['edge_index', 'edge_attr']]

    def raise_if_edge_keys(self):
        """This is a TEMPORARY, HACKY method to be called wherever
        edge_keys may cause an issue.
        """
        if len(self.edge_keys) > 0:
            raise NotImplementedError(
                "Edge keys are not fully supported yet, please consider "
                "stacking all your `edge_` attributes in `edge_attr` for the "
                "time being. This error was triggered by the presence of the "
                f"following attributes: {self.edge_keys}")

    @property
    def v_edge_keys(self) -> List[str]:
        """All keys starting with `v_edge_`."""
        return [k for k in self.keys if k.startswith('v_edge_')]

    @property
    def num_points(self):
        return self.num_nodes

    @property
    def num_super(self):
        return self.super_index.max().item() + 1 if self.is_sub else 0

    @property
    def num_sub(self):
        return self.sub.points.max().item() + 1 if self.is_super else 0

    @classmethod
    def from_flat_tensor(
            cls,
            tensors_dict: Dict[Any, torch.Tensor],
            flat_dict: Dict
    ) -> 'Data':
        """Reconstruct a flattened `Data` object from tensors as
        produced by `to_flat_tensor()`.

        :param tensors_dict: Dict[Any, Tensor]
            Dictionary of flat 1D tensors created by the
            `to_flat_tensor` utility function
        :param flat_dict: Dict
            Dictionary holding metadata for reconstructing the object
            from `tensors_dict`
        """
        return cls(**from_flat_tensor(tensors_dict, flat_dict))

    def forget_batching(self):
        """Change the class of the object to Data and remove any
        attributes characterizing Batch objects.
        """
        ignored_keys = [
            'batch',
            'ptr',
            '_slice_dict',
            '_inc_dict',
            '_num_graphs']
        self_dict = {
            k: v for k, v in self.to_dict().items()
            if k not in ignored_keys}
        for k, v in self_dict.items():
            if (hasattr(v.__class__, 'forget_batching')
                  and callable(v.__class__.forget_batching)):
                self_dict[k] = v.forget_batching()
        return Data(**self_dict)

    @property
    def device(self) -> torch.device:
        """Device of the first-encountered tensor in 'self'."""
        for key, item in self:
            if torch.is_tensor(item):
                return item.device
        return torch.tensor([]).device

    def debug(self):
        """Sanity checks."""
        self.validate()

        if self.is_super:
            assert isinstance(self.sub, Cluster), \
                "Clusters in 'sub' must be expressed using a Cluster object"
            assert self.y is None or self.y.dim() == 2, \
                "Clusters in 'sub' must hold label histograms"

        if self.obj is not None:
            assert isinstance(self.obj, InstanceData), \
                "Instance labels in 'obj' must be expressed using an " \
                "InstanceData object"

        if self.is_sub:
            if not is_dense(self.super_index):
                print(
                    "WARNING: super_index indices are generally expected to be "
                    "dense (i.e. all indices in [0, super_index.max()] are used),"
                    " which is not the case here. This may be because you are "
                    "creating a Data object after applying a selection of "
                    "points without updating the cluster indices.")

        if self.has_edges:
            assert self.edge_index.max() < self.num_points
            assert 0 <= self.edge_index.min()

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        """Extend the PyG.Data.__inc__ behavior on '*index*' and
        'face' attributes to our 'super_index'. This is needed for
        maintaining clusters when batching Data objects together.
        """
        if 'super_index' in key:
            return self.num_super
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        """Extend the PyG.Data.__cat_dim__ behavior on '*index*' and
        'face' attributes to our 'neighbor_index'. This is needed for
        maintaining neighbors when batching Data objects together.
        """
        return 0 if key == 'neighbor_index' \
            else super().__cat_dim__(key, value, *args, **kwargs)

    # TODO: this closely aligns with PyG.Data.subgraph(). Can't we
    #  refactor this to better depend on existing PyG code ?
    def select(
            self,
            idx: Union[int, List[int], torch.Tensor, np.ndarray],
            update_sub: bool = True,
            update_super: bool = True
    ) -> Tuple['Data', Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, 'Cluster']]:
        """Returns a new Data with updated clusters, which indexes
        `self` using entries in `idx`. Supports torch and numpy fancy
        indexing. `idx` must not contain duplicate entries, as this
        would cause ambiguities in edges and super- and sub- indices.

        This operations breaks neighborhoods, so if 'self.has_neighbors'
        the output Data will not.

        NB: if `self` belongs to a NAG, calling this function in
        isolation may break compatibility with point and cluster indices
        in the other hierarchy levels. If consistency matters, prefer
        using `NAG.select()` indexing instead.

        :param idx: int or 1D torch.LongTensor or numpy.NDArray
            Data indices to select from 'self'. Must NOT contain
            duplicates
        :param update_sub: bool
            If True, the point (i.e. subpoint) indices will also be
            updated to maintain dense indices. The output will then
            contain '(idx_sub, sub_super)' which can help apply these
            changes to maintain consistency with lower hierarchy levels
            of a NAG
        :param update_super: bool
            If True, the cluster (i.e. superpoint) indices will also be
            updated to maintain dense indices. The output will then
            contain '(idx_super, super_sub)' which can help apply these
            changes to maintain consistency with higher hierarchy levels
            of a NAG

        :return: data, (idx_sub, sub_super), (idx_super, super_sub)
            data: Data
                indexed data
            idx_sub: torch.LongTensor
                to be used with 'Data.select()' on the sub-level
            sub_super: torch.LongTensor
                to replace 'Data.super_index' on the sub-level
            idx_super: torch.LongTensor
                to be used with 'Data.select()' on the super-level
            super_sub: Cluster
                to replace 'Data.sub' on the super-level
        """
        device = self.device

        # If the idx is None, or is exactly [0, ..., num_cluster], we
        # leave the Cluster unchanged. As a result, (idx_sub, sub_super)
        # will not be needed for updating the indices for lower
        # partition levels
        idx = tensor_idx(idx, device=device)
        no_indexing_required = idx is None or is_arange(idx, self.num_nodes)
        if no_indexing_required:
            return self.clone(), (None, None), (None, None)

        # Make sure idx contains no duplicate entries
        if src.is_debug_enabled():
            assert not has_duplicates(idx), \
                "Duplicate indices are not supported. This would cause " \
                "ambiguities in edges and super- and sub- indices."

        # Output Data will not share memory with input Data.
        # NB: it is generally not recommended to instantiate en empty
        # Data like this, as it might cause issues when calling
        # 'data.num_nodes' later on. Need to be careful when calling
        # 'data.num_nodes' before having set any of the pointwise
        # attributes (e.g. 'x', 'pos', 'rgb', 'y', etc.)
        data = Data()

        # If Data contains edges, we will want to update edge indices
        # and attributes with respect to the new point order. Edge
        # indices are updated here, to compute 'idx_edge', which
        # will be used to select edge attributes
        if self.has_edges:
            # To update edge indices, create a 'reindex' tensor so that
            # the desired output can be computed with simple indexation
            # 'reindex[edge_index]'. This avoids using map() or
            # numpy.vectorize alternatives.
            reindex = torch.full(
                (self.num_nodes,), -1, dtype=torch.int64, device=device)
            reindex = reindex.scatter_(
                0, idx, torch.arange(idx.shape[0], device=device))
            edge_index = reindex[self.edge_index.long()]

            # Remove obsolete edges (i.e. those involving a '-1' index)
            idx_edge = torch.where((edge_index != -1).all(dim=0))[0]
            data.edge_index = edge_index[:, idx_edge]

        # Selecting points may affect their order, if we need to
        # preserve subpoint consistency, we need to update the
        # 'Data.sub' of the current level and the 'Data.super_index'
        # of the level below
        out_sub = (None, None)
        if self.is_super:
            if isinstance(self.sub, CSRData):
                data.sub, out_sub = self.sub.select(idx, update_sub=update_sub)
            else:
                # TODO: provide support for indexing when 'sub' is only
                #  a Tensor... Really an edge case, no priority
                data.sub = self.sub[idx]
                if update_sub:
                    raise NotImplementedError()

        # Selecting points may affect their order, if we need to
        # preserve superpoint consistency, we need to update the
        # 'Data.super_index' of the current level along with the
        # 'Data.sub' of the level above
        out_super = (None, None)
        if self.is_sub:
            data.super_index = self.super_index[idx]

        if self.is_sub and update_super:
            # Convert superpoint indices, in case some superpoints have
            # disappeared. 'idx_super' is intended to be used with
            # Data.select() on the level above
            new_super_index, perm = consecutive_cluster(data.super_index)
            idx_super = data.super_index[perm]
            data.super_index = new_super_index

            # Selecting the superpoints with 'idx_super' will not be
            # enough to maintain consistency with the current points. We
            # also need to update the super-level's 'Data.sub', which
            # can be computed from 'super_index'
            super_sub = Cluster(
                data.super_index,
                torch.arange(idx.shape[0], device=device),
                dense=True)

            out_super = (idx_super, super_sub)

        # Index data items depending on their type
        warn_keys = ['neighbor_index', 'neighbor_distance']
        skip_keys = ['edge_index', 'sub', 'super_index'] + warn_keys
        for key, item in self:

            # 'skip_keys' have already been dealt with earlier on, so we
            # can skip them here
            if key in warn_keys and src.is_debug_enabled():
                print(
                    f"WARNING: Data.select does not support '{key}', this "
                    f"attribute will be absent from the output")
            if key in skip_keys:
                continue

            # Slice CSRData elements, unless specified otherwise
            if isinstance(item, CSRData):
                data[key] = item.select(idx)
                continue

            is_tensor = torch.is_tensor(item)
            is_node_size = item.shape[0] == self.num_nodes
            is_edge_size = item.shape[0] == self.num_edges

            # Slice tensor elements containing num_edges elements. Note
            # we deal with edges first, to rule out the case where
            # num_edges = num_nodes. This will deal with `edge_attr` but
            # also any other attribute whose key starts with 'edge_' and
            # whose first dimension size matches the number of edges in
            # `edge_index`. An exception is made for attributes
            # starting with 'v_edge': those are expected to be node
            # attributes and must be treated as such
            if is_tensor and is_node_size and key in self.v_edge_keys:
                data[key] = item[idx]

            elif self.has_edges and is_tensor and is_edge_size and \
                    key in ['edge_attr'] + self.edge_keys:
                data[key] = item[idx_edge]

            # Slice other tensor elements containing num_nodes elements
            elif is_tensor and is_node_size:
                data[key] = item[idx]

            # Other Data attributes are simply copied
            else:
                data[key] = copy.deepcopy(item)

        # Security just in case no node-level attribute was passed, Data
        # will not be able to properly infer its number of nodes
        if data.num_nodes != idx.shape[0]:
            data.num_nodes = idx.shape[0]

        return data, out_sub, out_super

    def is_isolated(self):
        """If self.has_edges, returns a boolean tensor of size
        self.num_nodes indicating which are absent from self.edge_index.
        Will raise an error if self.has_edges is False.
        """
        edge_index = self.edge_index if self.has_edges \
            else torch.zeros(2, 0, dtype=torch.long, device=self.device)
        return isolated_nodes(edge_index, num_nodes=self.num_nodes)

    def connect_isolated(self, k: int = 1) -> 'Data':
        """Search for nodes with no edges in the graph and connect them
        to their k nearest neighbors. Update self.edge_index and
        self.edge_attr accordingly.

        Will raise an error if self has no edges or no pos.

        Returns self updated with the newly-created edges.
        """
        assert self.pos is not None

        # Make sure there is no edge_attr if there is no edge_index
        if not self.has_edges:
            self.edge_attr = None

        self.raise_if_edge_keys()

        # Search for isolated nodes and exit if no node is isolated
        is_isolated = self.is_isolated()
        is_out = torch.where(is_isolated)[0]
        if not is_isolated.any():
            return self

        # Search the nearest nodes for isolated nodes, among all nodes
        # NB: we remove the nodes themselves from their own neighborhood
        high = self.pos.max(dim=0).values
        low = self.pos.min(dim=0).values
        r_max = (high - low).norm()
        neighbors, distances = knn_2(
            self.pos,
            self.pos[is_out],
            k + 1,
            r_max=r_max,
            batch_search=self.batch,
            batch_query=self.batch[is_out] if self.batch is not None else None)
        distances = distances[:, 1:]
        neighbors = neighbors[:, 1:]

        # Add new edges between the nodes
        source = is_out.repeat_interleave(k)
        target = neighbors.flatten()
        edge_index_new = torch.vstack((source, target))
        edge_index_old = self.edge_index
        self.edge_index = torch.cat((edge_index_old, edge_index_new), dim=1)

        # Exit here if there are no edge attributes
        if self.edge_attr is None:
            return self

        # If the edges have attributes, we also create attributes for
        # the new edges. There is no trivial way of doing so, the
        # heuristic here simply attempts to linearly regress the edge
        # weights based on the corresponding node distances.
        # First, get existing edges attributes and associated distance
        w = self.edge_attr
        s = edge_index_old[0]
        t = edge_index_old[1]
        d = (self.pos[s] - self.pos[t]).norm(dim=1)
        d_1 = torch.vstack((d, torch.ones_like(d))).T

        # Least square on d_1.x = w  (i.e. d.a + b = w)
        # NB: CUDA may crash trying to solve this simple system, in
        # which case we will fall back to CPU. Not ideal though
        try:
            a, b = torch.linalg.lstsq(d_1, w).solution
        except:
            if src.is_debug_enabled():
                print(
                    '\nWarning: torch.linalg.lstsq failed, trying again '
                    'on CPU')
            a, b = torch.linalg.lstsq(d_1.cpu(), w.cpu()).solution
            a = a.to(self.device)
            b = b.to(self.device)

        # Heuristic: linear approximation of w by d
        edge_attr_new = distances.flatten() * a + b

        # Append to existing self.edge_attr
        self.edge_attr = torch.cat((self.edge_attr, edge_attr_new))

        return self

    def to_trimmed(self, reduce: str = 'mean') -> 'Data':
        """Convert to 'trimmed' graph: same as coalescing with the
        additional constraint that (i, j) and (j, i) edges are duplicates.

        If edge attributes are passed, 'reduce' will indicate how to fuse
        duplicate edges' attributes.

        NB: returned edges are expressed with i<j by default.
        """
        assert self.has_edges

        self.raise_if_edge_keys()

        if self.edge_attr is not None:
            edge_index, edge_attr = to_trimmed(
                self.edge_index, edge_attr=self.edge_attr, reduce=reduce)
        else:
            edge_index = to_trimmed(self.edge_index)
            edge_attr = None

        self.edge_index = edge_index
        self.edge_attr = edge_attr

        return self

    def equal(self, other: Any) -> bool:
        if (not isinstance(other, self.__class__)
                and not isinstance(self, other.__class__)):
            if src.is_debug_enabled():
                print(
                    f"{self.__class__.__name__}.equal: classes differ: "
                    f"{self.__class__.__name__} and {other.__class__.__name__}")
            return False
        if sorted(self.keys) != sorted(other.keys):
            if src.is_debug_enabled():
                print(f"{self.__class__.__name__}.equal: keys differ")
            return False
        for k, v in self.items():
            msg = f"{self.__class__.__name__}.equal: {k} differ"
            if isinstance(v, torch.Tensor):
                if torch.equal(v, other[k]):
                    continue
                if src.is_debug_enabled():
                    print(msg)
                return False
            if isinstance(v, np.ndarray):
                if np.array_equal(v, other[k]):
                    continue
                if src.is_debug_enabled():
                    print(msg)
                return False
            if isinstance(v, CSRData):
                if v.equal(other[k]):
                    continue
                if src.is_debug_enabled():
                    print(msg)
                return False
            if v != other[k]:
                if src.is_debug_enabled():
                    print(msg)
                return False
        return True

    @property
    def nbytes(self):
        """Returns the number of bytes consumed by all the Tensors held
         in the object.

         Specifically, this is the sum `tensor.nbytes` for al Tensors in
        the object.
        """
        nbytes = 0
        for _, item in self.items():
            if torch.is_tensor(item):
                nbytes += item.nbytes
                continue
            try:
                nbytes += item.nbytes
            except AttributeError:
                pass
        return nbytes

    def print_memory_summary(self, depth=0, indent=2, prefix=''):
        """Print a summary of the memory usage for tensors inside the
        object. This is a detailed breakdown of what `self.nbytes`
        returns.
        """
        print(f"{prefix} {self.__class__.__name__}: {human_readable_memory(self.nbytes)}")
        white_length = max([len(str(key)) for key, _ in self.items()])
        for key, item in self.to_dict().items():
            prefix_key = f"{'' * len(prefix)}{' ' * indent * (depth + 1)} {key:<{white_length}}"
            if torch.is_tensor(item):
                print(f"{prefix_key}: {human_readable_memory(item.nbytes)}  {tuple(item.shape)}  {item.dtype}")
                continue
            try:
                item.print_memory_summary(
                    depth=depth + 1, indent=indent, prefix=f"{prefix_key}")
            except AttributeError:
                print(f"{prefix_key}: ignored")

    def save(
            self,
            f: Union[str, h5py.File, h5py.Group],
            y_to_csr: bool = True,
            pos_dtype: torch.dtype = torch.float,
            fp_dtype: torch.dtype = torch.float,
            rgb_to_byte: bool = True):
        """Save Data to HDF5 file.

        :param f: h5 file path of h5py.File or h5py.Group
        :param y_to_csr: bool
            Convert 'y' to CSR format before saving. Only applies if
            'y' is a 2D histogram
        :param pos_dtype: torch dtype
            Data type to which 'pos' should be cast before saving. The
            reason for this separate treatment of 'pos' is that global
            coordinates may be too large and casting to 'fp_dtype' may
            result in hurtful precision loss
        :param fp_dtype: torch dtype
            Data type to which floating point tensors should be cast
            before saving
        :param rgb_to_byte: bool
            Whether to cast the 'rgb' and 'mean_rgb' attributes to byte
            (uint8) before saving.
        :return:
        """
        if not isinstance(f, (h5py.File, h5py.Group)):
            with h5py.File(f, 'w') as file:
                self.save(
                    file,
                    y_to_csr=y_to_csr,
                    pos_dtype=pos_dtype,
                    fp_dtype=fp_dtype,
                    rgb_to_byte=rgb_to_byte)
            return

        assert isinstance(f, (h5py.File, h5py.Group))

        for k, val in self.items():
            if k == 'pos_offset':
                save_tensor(val, f, k, fp_dtype=torch.double)
            elif k == 'pos':
                save_tensor(val, f, k, fp_dtype=pos_dtype)
            elif k == 'y' and val.dim() > 1 and y_to_csr:
                sg = f.create_group(f"{f.name}/_csr_/{k}")
                save_dense_to_csr(val, sg, fp_dtype=fp_dtype)
            elif k in ['rgb', 'mean_rgb'] and rgb_to_byte:
                if val.is_floating_point():
                    save_tensor((val * 255).byte(), f, k, fp_dtype=fp_dtype)
                else:
                    save_tensor(val.byte(), f, k, fp_dtype=fp_dtype)
            elif isinstance(val, Cluster):
                sg = f.create_group(f"{f.name}/_cluster_/{k}")
                val.save(sg, fp_dtype=fp_dtype)
            elif isinstance(val, InstanceData):
                sg = f.create_group(f"{f.name}/_instance_data_/{k}")
                val.save(sg, fp_dtype=fp_dtype)
            elif isinstance(val, CSRData):
                sg = f.create_group(f"{f.name}/_csr_/{k}")
                val.save(sg, fp_dtype=fp_dtype)
            elif isinstance(val, torch.Tensor):
                save_tensor(val, f, k, fp_dtype=fp_dtype)
            else:
                raise NotImplementedError(
                    f"Cannot save attribute {k} with unsupported type "
                    f"{type(val)}")

        # Save the information of which attributes are indexable and
        # which are not
        f['_not_indexable_'] = list(set(self.keys) - set(self.node_attrs()))

    @classmethod
    def load(
            cls,
            f: Union[h5py.File, h5py.Group],
            idx: Union[int, List, np.ndarray, torch.Tensor] = None,
            keys_idx: List[str] = None,
            keys: List[str] = None,
            update_sub: bool = True,
            non_fp_to_long: bool = False,
            verbose: bool = False,
            rgb_to_float: bool = False
    ) -> 'Data':
        """Read an HDF5 file and return its content as a Data object.

        NB: if relevant, a Batch object will be returned.

        :param f: h5 file path of h5py.File or h5py.Group
        :param idx: int, list, numpy.ndarray, torch.Tensor
            Used to select the elements in `keys_idx`. Supports fancy
            indexing
        :param keys_idx: List(str)
            Keys on which the indexing should be applied
        :param keys: List(str)
            Keys should be loaded from the file, ignoring the rest
        :param update_sub: bool
            If True, the point (i.e. subpoint) indices will also be
            updated to maintain dense indices. The output will then
            contain '(idx_sub, sub_super)' which can help apply these
            changes to maintain consistency with lower hierarchy levels
            of a NAG.
        :param non_fp_to_long: bool
            By default `save_tensor()` cast all non-float tensors to the
            smallest integer dtype before saving. This allows saving
            memory and I/O bandwidth. Upon reading, `non_fp_to_long`
            rules whether these should be cast back to int64 or kept in
            this "compressed" dtype. One good reason for not doing so is
            to accelerate data loading and device transfer. To cast the
            tensors to int64 later on in the pipeline, use the `NAGCast`
            and `Cast` transforms
        :param verbose: bool
        :param rgb_to_float: bool
            If True and an integer 'rgb' or 'mean_rgb' attribute is
            loaded, it will be cast to float
        :return:
        """
        if not isinstance(f, (h5py.File, h5py.Group)):
            with h5py.File(f, 'r') as file:
                out = cls.load(
                    file,
                    idx=idx,
                    keys_idx=keys_idx,
                    keys=keys,
                    update_sub=update_sub,
                    non_fp_to_long=non_fp_to_long,
                    verbose=verbose,
                    rgb_to_float=rgb_to_float)
            return out

        # Recover the keys that do not support node indexing
        _not_indexable = cls._NOT_INDEXABLE
        if '_not_indexable_' in f.keys():
            _not_indexable += [s.decode("utf-8") for s in f['_not_indexable_']]

        # Check whether the indexing is actually effective. If not, all
        # indexing-related behavior can be skipped for simplicity
        # TODO: if 'pos' is not in the saved file keys, need a backup to
        #  infer the number of nodes
        idx = tensor_idx(idx)
        num_nodes = f['pos'].shape[0] - 1
        no_indexing_required = idx is None or is_arange(idx, num_nodes)

        if no_indexing_required:
            idx = None
            keys_idx = []
        elif keys_idx is None:
            keys_idx = list(set(f.keys()) - set(_not_indexable))

        if keys is None:
            all_keys = list(f.keys())
            for k in ['_csr_', '_cluster_', '_instance_data_']:
                if k in all_keys:
                    all_keys.remove(k)
                    all_keys += list(f[k].keys())
            keys = all_keys

        d_dict = {}
        csr_keys = []
        cluster_keys = []
        instance_data_keys = []

        # Check if the file actually corresponds to a Batch object
        # rather than a simple Data object
        has_slice_and_inc = '_slice_dict' in f.keys() and '_inc_dict' in f.keys()
        has_batch = 'batch' in f.keys()
        if (has_slice_and_inc or has_batch) and no_indexing_required:
            cls = Batch
        else:
            cls = Data

        # Deal with special keys first, then read other keys if required
        for k in f.keys():
            start = time()
            if k == '_not_indexable_':
                continue
            if k == '_csr_':
                csr_keys = list(f[k].keys())
                continue
            if k == '_cluster_':
                cluster_keys = list(f[k].keys())
                continue
            if k == '_instance_data_':
                instance_data_keys = list(f[k].keys())
                continue
            if k in ['_slice_dict', '_inc_dict']:
                if cls == Batch:
                    d_dict[k] = load_tensor_dict(
                        f[k], non_fp_to_long=non_fp_to_long)
                continue
            if k == '_num_graphs':
                if cls == Batch:
                    d_dict[k] = f['_num_graphs'][0]
                continue
            if k in keys_idx:
                d_dict[k] = load_tensor(f[k], idx=idx, non_fp_to_long=non_fp_to_long)
            elif k in keys:
                d_dict[k] = load_tensor(f[k], non_fp_to_long=non_fp_to_long)
            if verbose and k in d_dict.keys():
                print(f'{cls.__name__}.load {k:<22}: {time() - start:0.5f}s')

        # Small sanity check on '_slice_dict' and '_inc_dict'. It is
        # possible, for attributes of type 'other' that the _inc_dict
        # contains a 'None' value when the _slice_dict holds a Tensor.
        # As a result of calling 'save_tensor_dict()' and
        # 'load_tensor_dict()', these values will be lost. So we need to
        # restore them here
        if '_slice_dict' in d_dict.keys():
            for k in set(d_dict['_slice_dict']) - set(d_dict['_inc_dict']):
                d_dict['_inc_dict'][k] = None

        # Update the 'keys_idx' with newly-found 'csr_keys',
        # 'cluster_keys', and 'instance_data_keys'
        if not no_indexing_required:
            keys_idx = list(set(
                keys_idx + csr_keys + cluster_keys + instance_data_keys))

        # Special key '_csr_' holds data saved in CSR format
        for k in csr_keys:
            start = time()
            if k in keys_idx:
                d_dict[k] = load_csr_to_dense(
                    f['_csr_'][k],
                    idx=idx,
                    non_fp_to_long=non_fp_to_long,
                    verbose=verbose)
            elif k in keys:
                d_dict[k] = load_csr_to_dense(
                    f['_csr_'][k],
                    non_fp_to_long=non_fp_to_long,
                    verbose=verbose)
            if verbose and k in d_dict.keys():
                print(f'{cls.__name__}.load {k:<22}: {time() - start:0.5f}s')

        # Special key '_cluster_' holds Cluster data
        for k in cluster_keys:
            start = time()
            if k in keys_idx:
                d_dict[k] = Cluster.load(
                    f['_cluster_'][k],
                    idx=idx,
                    update_sub=update_sub,
                    non_fp_to_long=non_fp_to_long,
                    verbose=verbose)[0]
            elif k in keys:
                d_dict[k] = Cluster.load(
                    f['_cluster_'][k],
                    update_sub=update_sub,
                    non_fp_to_long=non_fp_to_long,
                    verbose=verbose)[0]
            if verbose and k in d_dict.keys():
                print(f'{cls.__name__}.load {k:<22}: {time() - start:0.5f}s')

        # Special key '_instance_data_' holds InstanceData data
        for k in instance_data_keys:
            start = time()
            if k in keys_idx:
                d_dict[k] = InstanceData.load(
                    f['_instance_data_'][k],
                    idx=idx,
                    non_fp_to_long=non_fp_to_long,
                    verbose=verbose)
            elif k in keys:
                d_dict[k] = InstanceData.load(
                    f['_instance_data_'][k],
                    non_fp_to_long=non_fp_to_long,
                    verbose=verbose)
            if verbose and k in d_dict.keys():
                print(f'{cls.__name__}.load {k:<22}: {time() - start:0.5f}s')

        # In case RGB is among the keys and is in integer type, convert
        # to float
        for k in ['rgb', 'mean_rgb']:
            if k in d_dict.keys():
                d_dict[k] = to_float_rgb(d_dict[k]) if rgb_to_float \
                    else to_byte_rgb(d_dict[k])

        return cls(**d_dict)

    def estimate_instance_centroid(
            self,
            mode: str = 'iou'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate the centroid position of each target instance
        object, based on the position of the clusters.

        Based on the hypothesis that clusters are relatively
        instance-pure, we approximate the centroid of each object by
        taking the barycenter of the centroids of the clusters
        overlapping with each object, weighed down by their respective
        IoUs.

        NB: This is a proxy and one could design failure cases, when
        clusters are not pure enough.

        :param mode: str
            Method used to estimate the centroids. 'iou' will weigh down
            the centroids of the clusters overlapping each instance by
            their IoU. 'ratio-product' will use the product of the size
            ratios of the overlap wrt the cluster and wrt the instance.
            'overlap' will use the size of the overlap between the
            cluster and the instance.

        :return obj_pos, obj_idx
            obj_pos: Tensor
                Estimated position for each object
            obj_idx: Tensor
                Corresponding object indices
        """
        if self.obj is None:
            return None, None

        return self.obj.estimate_centroid(self.pos, mode=mode)

    def semantic_segmentation_oracle(
            self,
            num_classes: int,
            *metric_args,
            **metric_kwargs
    ) -> SemanticMetricResults:
        """Compute the oracle performance for semantic segmentation,
        when all nodes predict the dominant label among their points.
        This corresponds to the highest achievable performance with the
        partition at hand.

        This expects one of the following attributes:
          - `Data.obj`: holding node overlaps with instance annotations
          - `Data.y`: holding node label histograms

        :param num_classes: int
            Number of valid classes. By convention, we assume
            `y ∈ [0, num_classes-1]` are VALID LABELS, while
            `y < 0` AND `y >= num_classes` ARE VOID LABELS
        :param metric_args:
            Args for the metrics computation
        :param metric_kwargs:
            Kwargs for the metrics computation

        :return: mIoU, pre-class IoU, OA, mAcc
        """
        # Rely on the InstanceData for computation, if any
        if self.obj is not None:
            return self.obj.semantic_segmentation_oracle(
                num_classes, *metric_args, **metric_kwargs)

        # Return None if no labels
        if getattr(self, 'y', None) is None:
            return

        # We expect the network to predict the most frequent label. For
        # clusters where the dominant label is 'void', we expect the
        # network to predict the second most frequent label. In the
        # event where the cluster is 100% 'void', the metric will ignore
        # the prediction, regardless its value
        pred = self.y[:, :num_classes].argmax(dim=1)
        target = self.y

        # Performance evaluation
        from src.metrics import ConfusionMatrix
        cm = ConfusionMatrix(num_classes, *metric_args, **metric_kwargs)
        cm(pred.cpu(), target.cpu())
        metrics = cm.all_metrics()

        return metrics

    def instance_segmentation_oracle(
            self,
            *metric_args,
            **metric_kwargs
    ) -> InstanceMetricResults:
        """Compute the oracle performance for instance segmentation.
        This is a proxy for the highest achievable performance with the
        cluster partition at hand.

        More precisely, for the oracle prediction:
          - each cluster is assigned to the instance it shares the most
            points with
          - clusters assigned to the same instance are merged into a
            single prediction
          - each predicted instance has a score equal to its IoU with
            the assigned target instance

        This expects the following attributes:
          - `Data.obj`: holding node overlaps with instance annotations

        :param metric_args:
            Args for the metrics computation
        :param metric_kwargs:
            Kwargs for the metrics computation

        :return: InstanceMetricResults
        """
        # Rely on the InstanceData for computation, if any
        if self.obj is not None:
            return self.obj.instance_segmentation_oracle(
                *metric_args, **metric_kwargs)
        return

    def panoptic_segmentation_oracle(
            self,
            *metric_args,
            **metric_kwargs
    ) -> PanopticMetricResults:
        """Compute the oracle performance for panoptic segmentation.
        This is a proxy for the highest achievable performance with the
        cluster partition at hand.

        More precisely, for the oracle prediction:
          - each cluster is assigned to the instance it shares the most
            points with
          - clusters assigned to the same instance are merged into a
            single prediction

        This expects the following attributes:
          - `Data.obj`: holding node overlaps with instance annotations

        :param metric_args:
            Args for the metrics computation
        :param metric_kwargs:
            Kwargs for the metrics computation

        :return: PanopticMetricResults
        """
        # Rely on the InstanceData for computation, if any
        if self.obj is not None:
            return self.obj.panoptic_segmentation_oracle(
                *metric_args, **metric_kwargs)
        return

    def show(self, **kwargs):
        """See `src.visualization.show`."""
        # Local import to avoid import loop errors
        from src.visualization import show
        return show(self, **kwargs)

    def add_keys_to(
            self,
            keys: List[str],
            to: str = 'x',
            strict: bool = True,
            delete_after: bool = False
    ) -> None:
        """Add node features to the Data object."""
        if keys is None or len(keys) == 0:
            return
        
        previous_feat = getattr(self, to, None)
        
        # Initialize the list that will contain all the keys to be concatenated
        if previous_feat is not None:
            feat_list = [previous_feat]
        else:
            feat_list = []
        
        # Loop on the keys to add
        for key in keys:
            # Skip if the attribute is None
            feat = getattr(self, key, None)
            if feat is None:
                if strict:
                    raise Exception(f"Data should contain the attribute '{key}'")
                else:
                    continue
            if delete_after:
                delattr(self, key)
            
            if previous_feat is None:
                if strict and self.num_nodes != feat.shape[0]:
                    raise Exception(f"Data should contain the attribute '{to}'")
            elif previous_feat.shape[0] != feat.shape[0]: # Make sure shapes match
                raise Exception(
                    f"The tensors '{to}' and '{key}' can't be concatenated, "
                    f"'{to}': {previous_feat.shape[0]}, '{key}': {feat.shape[0]}")
            
            if feat.dim() == 1:
                feat = feat.unsqueeze(-1)

            feat_list.append(feat)

        setattr(self, to, torch.cat(feat_list, dim=1))
