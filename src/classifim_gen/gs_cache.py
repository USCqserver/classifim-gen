"""
This module implements a cache for ground state computations.
"""
import classifim_utils
import enum
import filelock
import numpy as np
import os.path

class GroundStateComputationError(Exception):
    def __init__(self, *args, failure_dict=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.failure_dict = failure_dict or {}
    def __str__(self):
        return (
            f"{super().__str__()}\n" +
            "failure_dict keys: (" + ", ".join(self.failure_dict.keys()) + ")")

class FilenameSource(enum.Enum):
    # META:
    # - Fail if file linked from meta is missing
    # - Use if you want to trust the meta file
    META = 1

    # FILESYSTEM:
    # - Ignore meta contents (but update it based on filesystem)
    # - Use if you want to use cached files copied from elsewhere.
    FILESYSTEM = 2

    # Fail on inconsistency
    # - Most strict. Fails if meta file is inconsistent with filesystem.
    META_AND_FILESYSTEM = 3

FS_META = FilenameSource.META
FS_FILESYSTEM = FilenameSource.FILESYSTEM
FS_META_AND_FILESYSTEM = FilenameSource.META_AND_FILESYSTEM

class GroundStateCache:
    def __init__(self, compute_ground_state, param_keys, ham_name,
                 save_path, load_meta=None,
                 filename_source=FS_META_AND_FILESYSTEM):
        """
        This class caches the results of ground state computation in a directory save_path.

        Meta file structure:
            {save_path}/{ham_name}.gs_meta.npz contains metadata saved by this class:
            - ham_name
            - param_keys
            - save_path
            - params_vecs: shape=(num_runs, len(param_keys)), dtype=np.float64
            - file_paths: shape=(num_runs,) paths to individual files relative
            to save_path matching params_vecs.

        Args:
            compute_ground_state: function that takes a vector of parameters and returns
                a dict describing the ground state.
            param_keys: tuple of strings describing the parameters.
            ham_name: string describing the Hamiltonian family (used for path prefixes).
            save_path: string path to the directory where the cache is stored.
            load_meta: whether to attempt loading the meta file.
                If None, use (filename_source != FS_FILESYSTEM).
            filename_source: whether to use the meta file or the filesystem as the
                source of truth for the state of cache.
        """
        self.compute_ground_state = compute_ground_state
        self.param_keys = param_keys
        self.sorted_param_keys = sorted(self.param_keys)
        self.ham_name = ham_name
        self.save_path = save_path
        self.meta_file_name = f"{save_path}/{ham_name}.gs_meta.npz"
        self.filename_source = filename_source
        if load_meta is None:
            load_meta = (filename_source != FS_FILESYSTEM)
        self.load_meta = load_meta
        assert os.path.isdir(save_path), (
            f"'{save_path}' is not a directory")
        if self.load_meta and os.path.exists(self.meta_file_name):
            meta_npz = np.load(self.meta_file_name)
            assert np.array_equal(self.ham_name, meta_npz["ham_name"])
            assert np.array_equal(self.param_keys, meta_npz["param_keys"])
            params_vecs_key = "params_vecs"
            if params_vecs_key not in meta_npz:
                params_vecs_key = "param_values"
                assert params_vecs_key in meta_npz
            params_vecs = [tuple(p) for p in meta_npz[params_vecs_key]]
            assert len(params_vecs) == len(meta_npz["file_paths"])
            self.params_vec_to_file_paths = dict(
                    zip(params_vecs, meta_npz["file_paths"]))
        else:
            self.params_vec_to_file_paths = {}

    # TODO:9:remove
    # @staticmethod
    # def hash_base36(v, length=13):
    #     # TODO:9: hash(v) is not guaranteed to be stable.
    #     return np.base_repr(hash(v), 36, length)[-length:].lower()

    def get_gs_filename(self, params_vec, suffix=''):
        """
        Computes full filename to save the file with ground state data.

        Args:
            params_vec: values given in the same order as param_keys.
        """
        # Ensure that the data format is consistent, otherwise
        # hash may be incorrect.
        assert isinstance(params_vec, tuple)
        assert len(params_vec) == len(self.param_keys)
        for p in params_vec:
            assert isinstance(p, float)
        hash_str = classifim_utils.hash_base36(params_vec)
        return os.path.join(
            self.save_path,
            f"{self.ham_name}_{hash_str}{suffix}.npz")

    def cache_exists(self, params_vec, cache_type=""):
        """
        Checks if the ground state computation for the given parameters is cached.

        Args:
            params_vec: tuple with parameter values
            cache_type: one of the following:
                - "" (default): check if the ground state is cached
                - "failure": check if the ground state computation failed in the past
                - "lock": check if the ground state computation is currently running
                (or lock file is left from a previous run)
                - "any": check if any of the above is true
        """
        if cache_type == "":
            file_name = self.get_gs_filename(params_vec)
        elif cache_type == "failure":
            file_name = self.get_gs_filename(params_vec, suffix=".failure")
        elif cache_type == "lock":
            file_name = self.get_gs_filename(params_vec) + ".lock"
        elif cache_type == "any":
            return (
                self.cache_exists(params_vec, cache_type="") or
                self.cache_exists(params_vec, cache_type="failure") or
                self.cache_exists(params_vec, cache_type="lock"))
        else:
            raise ValueError(f"Unknown cache_type: {cache_type}")
        return os.path.exists(file_name)

    def params_tuple_to_dict(self, params_tuple):
        return dict(zip(self.param_keys, params_tuple))

    @staticmethod
    def _save_and_log(file_name, params_vec, data, verbose):
        np.savez_compressed(file_name, params=params_vec, **data)
        if verbose:
            time_str = ""
            if "time" in data:
                time_str = f" in {data['time']:.2f}s"
            print(f"Wrote '{file_name}' for {params_vec=}{time_str}.")

    @staticmethod
    def _find_filename_source(
            file_name, filename_from_meta, filename_source):
        if filename_source == FS_FILESYSTEM:
            return file_name, os.path.exists(file_name)
        if filename_source == FS_META:
            if filename_from_meta is None:
                return file_name, False
            assert os.path.exists(filename_from_meta)
            return filename_from_meta, True
        if filename_source == FS_META_AND_FILESYSTEM:
            if filename_from_meta is None:
                assert not os.path.exists(file_name), (
                    f"File '{file_name}' exists, but not in meta file.")
                return file_name, False
            if file_name != filename_from_meta:
                assert not os.path.exists(file_name), (
                    f"File at '{file_name}' is found, but meta says it "
                    f"should be at '{filename_from_meta}'.")
            assert os.path.exists(filename_from_meta), (
                    f"File '{filename_from_meta}' is not found, but meta "
                    f"says it should be there.")
            return filename_from_meta, True
        raise ValueError(f"Unknown filename_source: {filename_source}")

    def get_ground_state(self, params_vec, load=True, verbose=False):
        """
        Computes and saves or loads ground state computation.

        Args:
            params_vec: tuple with parameter values
            load: if False, skip loading (return None if already computed).
              Useful for ensuring the result of the computation is cached.

        Returns:
            The output of `self.compute_ground_state` (dict, if not cached),
            or the output of `np.load` (npz, if cached and `load == True`),
            or `None` (if cached and `load == False`).
        """
        file_name = self.get_gs_filename(params_vec)
        filename_from_meta = self.params_vec_to_file_paths.get(params_vec)
        if filename_from_meta is not None:
            filename_from_meta = os.path.join(self.save_path, filename_from_meta)
        file_name, can_load = self._find_filename_source(
            file_name, filename_from_meta, self.filename_source)
        rel_file_name = os.path.relpath(file_name, start=self.save_path)
        if can_load:
            self.params_vec_to_file_paths[params_vec] = rel_file_name
            if load:
                return np.load(file_name)
            else:
                return None
        with filelock.SoftFileLock(file_name + ".lock", timeout=1):
            try:
                res = self.compute_ground_state(params_vec)
                self._save_and_log(
                    file_name, params_vec, res, verbose)
            except GroundStateComputationError as e:
                failure_file_name = self.get_gs_filename(
                    params_vec, suffix='.failure')
                self._save_and_log(
                    failure_file_name, params_vec,
                    e.failure_dict, verbose)
                raise e
        self.params_vec_to_file_paths[params_vec] = rel_file_name
        return res

    def save_meta(self):
        np.savez_compressed(
            self.meta_file_name,
            ham_name=self.ham_name,
            param_keys=self.param_keys,
            save_path=self.save_path,
            about=f"Saved by GroundStateCache.save_meta function.",
            params_vecs=np.array(list(self.params_vec_to_file_paths.keys())),
            file_paths=np.array(list(self.params_vec_to_file_paths.values()))
        )

    def close(self):
        self.save_meta()

