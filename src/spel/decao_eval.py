"""
This file serves as a standalone evaluation provider for evaluating the predictions of a entity linking system.
 The content of this module are taken from https://github.com/nicola-decao/efficient-autoregressive-EL and the necessary
  boilerplate code is copied along with the metric classes to help the code act as standalone.

To perform evaluation, import the following classes (or any subset of the evaluation metrics that you need):
            MicroF1, MicroPrecision, MicroRecall, MacroRecall, MacroPrecision, MacroF1
Collect the el_model predictions in the format of {(start_index, end_index, annotation string)} for document d.
Collect the gold dataset annotations in the format of {(start_index, end_index, annotation string)} for document d.
Call the metric instances for the two mentioned sets p and g:
    micro_f1(p, g)
    micro_prec(p, g)
    micro_rec(p, g)
    macro_f1(p, g)
    macro_prec(p, g)
    macro_rec(p, g)

Once you are done with all the documents and all predictions are added, you may access the evaluation results using:
    {'macro_f1': macro_f1.compute(),
     'macro_prec': macro_prec.compute(),
     'macro_rec': macro_rec.compute(),
     'micro_f1': micro_f1.compute(),
     'micro_prec': micro_prec.compute(),
     'micro_rec': micro_rec.compute()}
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Hashable, Iterable, Generator, Sequence, Tuple, Union, List, Mapping, Callable, Optional
import operator as op
import functools
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from contextlib import contextmanager
import inspect
from collections import OrderedDict
from copy import deepcopy
from importlib import import_module
from importlib.util import find_spec

from packaging.version import Version
from pkg_resources import DistributionNotFound, get_distribution


def dim_zero_sum(x: Tensor) -> Tensor:
    """summation along the zero dimension."""
    return torch.sum(x, dim=0)


def dim_zero_mean(x: Tensor) -> Tensor:
    """average along the zero dimension."""
    return torch.mean(x, dim=0)


def dim_zero_max(x: Tensor) -> Tensor:
    """max along the zero dimension."""
    return torch.max(x, dim=0).values


def dim_zero_min(x: Tensor) -> Tensor:
    """min along the zero dimension."""
    return torch.min(x, dim=0).values


def dim_zero_cat(x: Union[Tensor, List[Tensor]]) -> Tensor:
    """concatenation along the zero dimension."""
    x = x if isinstance(x, (list, tuple)) else [x]
    x = [y.unsqueeze(0) if y.numel() == 1 and y.ndim == 0 else y for y in x]
    if not x:  # empty list
        raise ValueError("No samples to concatenate")
    return torch.cat(x, dim=0)


def _module_available(module_path: str) -> bool:
    try:
        return find_spec(module_path) is not None
    except AttributeError:
        # Python 3.6
        return False
    except ModuleNotFoundError:
        # Python 3.7+
        return False


def _compare_version(package: str, op: Callable, version: str) -> Optional[bool]:
    if not _module_available(package):
        return None
    try:
        pkg = import_module(package)
        pkg_version = pkg.__version__  # type: ignore
    except (ModuleNotFoundError, DistributionNotFound):
        return None
    except ImportError:
        # catches cyclic imports - the case with integrated libs
        # see: https://stackoverflow.com/a/32965521
        pkg_version = get_distribution(package).version
    try:
        pkg_version = Version(pkg_version)
    except TypeError:
        # this is mock by sphinx, so it shall return True ro generate all summaries
        return True
    return op(pkg_version, Version(version))


class TorchMetricsUserError(Exception):
    """Error used to inform users of a wrong combinison of Metric API calls."""


def _simple_gather_all_tensors(result: Tensor, group: Any, world_size: int) -> List[Tensor]:
    gathered_result = [torch.zeros_like(result) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_result, result, group)
    return gathered_result


def gather_all_tensors(result: Tensor, group: Optional[Any] = None) -> List[Tensor]:
    """Function to gather all tensors from several ddp processes onto a list that is broadcasted to all processes.
    Works on tensors that have the same number of dimensions, but where each dimension may differ. In this case
    tensors are padded, gathered and then trimmed to secure equal workload for all processes.

    Args:
        result: the value to sync
        group: the process group to gather results from. Defaults to all processes (world)

    Return:
        gathered_result: list with size equal to the process group where
            gathered_result[i] corresponds to result tensor from process i
    """
    if group is None:
        group = torch.distributed.group.WORLD

    # convert tensors to contiguous format
    result = result.contiguous()

    world_size = torch.distributed.get_world_size(group)
    torch.distributed.barrier(group=group)

    # if the tensor is scalar, things are easy
    if result.ndim == 0:
        return _simple_gather_all_tensors(result, group, world_size)

    # 1. Gather sizes of all tensors
    local_size = torch.tensor(result.shape, device=result.device)
    local_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
    torch.distributed.all_gather(local_sizes, local_size, group=group)
    max_size = torch.stack(local_sizes).max(dim=0).values
    all_sizes_equal = all(all(ls == max_size) for ls in local_sizes)

    # 2. If shapes are all the same, then do a simple gather:
    if all_sizes_equal:
        return _simple_gather_all_tensors(result, group, world_size)

    # 3. If not, we need to pad each local tensor to maximum size, gather and then truncate
    pad_dims = []
    pad_by = (max_size - local_size).detach().cpu()
    for val in reversed(pad_by):
        pad_dims.append(0)
        pad_dims.append(val.item())
    result_padded = F.pad(result, pad_dims)
    gathered_result = [torch.zeros_like(result_padded) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_result, result_padded, group)
    for idx, item_size in enumerate(local_sizes):
        slice_param = [slice(dim_size) for dim_size in item_size]
        gathered_result[idx] = gathered_result[idx][slice_param]
    return gathered_result


def apply_to_collection(
        data: Any,
        dtype: Union[type, tuple],
        function: Callable,
        *args: Any,
        wrong_dtype: Optional[Union[type, tuple]] = None,
        **kwargs: Any,
) -> Any:
    """Recursively applies a function to all elements of a certain dtype.

    Args:
        data: the collection to apply the function to
        dtype: the given function will be applied to all elements of this dtype
        function: the function to apply
        *args: positional arguments (will be forwarded to calls of ``function``)
        wrong_dtype: the given function won't be applied if this type is specified and the given collections is of
            the :attr:`wrong_type` even if it is of type :attr`dtype`
        **kwargs: keyword arguments (will be forwarded to calls of ``function``)

    Returns:
        the resulting collection

    Example:
        >>> apply_to_collection(torch.tensor([8, 0, 2, 6, 7]), dtype=Tensor, function=lambda x: x ** 2)
        tensor([64,  0,  4, 36, 49])
        >>> apply_to_collection([8, 0, 2, 6, 7], dtype=int, function=lambda x: x ** 2)
        [64, 0, 4, 36, 49]
        >>> apply_to_collection(dict(abc=123), dtype=int, function=lambda x: x ** 2)
        {'abc': 15129}
    """
    elem_type = type(data)

    # Breaking condition
    if isinstance(data, dtype) and (wrong_dtype is None or not isinstance(data, wrong_dtype)):
        return function(data, *args, **kwargs)

    # Recursively apply to collection items
    if isinstance(data, Mapping):
        return elem_type({k: apply_to_collection(v, dtype, function, *args, **kwargs) for k, v in data.items()})

    if isinstance(data, tuple) and hasattr(data, "_fields"):  # named tuple
        return elem_type(*(apply_to_collection(d, dtype, function, *args, **kwargs) for d in data))

    if isinstance(data, Sequence) and not isinstance(data, str):
        return elem_type([apply_to_collection(d, dtype, function, *args, **kwargs) for d in data])

    # data is neither of dtype, nor a collection
    return data


def _flatten(x: Sequence) -> list:
    return [item for sublist in x for item in sublist]


def jit_distributed_available() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


class _Metric(nn.Module, ABC):
    __jit_ignored_attributes__ = ["device"]
    __jit_unused_properties__ = ["is_differentiable"]
    is_differentiable: Optional[bool] = None
    higher_is_better: Optional[bool] = None

    def __init__(
            self,
            compute_on_step: bool = True,
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
            dist_sync_fn: Callable = None,
    ) -> None:
        super().__init__()

        # see (https://github.com/pytorch/pytorch/blob/3e6bb5233f9ca2c5aa55d9cda22a7ee85439aa6e/
        # torch/nn/modules/module.py#L227)
        torch._C._log_api_usage_once(f"torchmetrics.metric.{self.__class__.__name__}")

        self._LIGHTNING_GREATER_EQUAL_1_3 = _compare_version("pytorch_lightning", op.ge, "1.3.0")
        self._device = torch.device("cpu")

        self.dist_sync_on_step = dist_sync_on_step
        self.compute_on_step = compute_on_step
        self.process_group = process_group
        self.dist_sync_fn = dist_sync_fn
        self._to_sync = True
        self._should_unsync = True

        self._update_signature = inspect.signature(self.update)
        self.update: Callable = self._wrap_update(self.update)  # type: ignore
        self.compute: Callable = self._wrap_compute(self.compute)  # type: ignore
        self._computed = None
        self._forward_cache = None
        self._update_called = False

        # initialize state
        self._defaults: Dict[str, Union[List, Tensor]] = {}
        self._persistent: Dict[str, bool] = {}
        self._reductions: Dict[str, Union[str, Callable[[Union[List[Tensor], Tensor]], Tensor], None]] = {}

        # state management
        self._is_synced = False
        self._cache: Optional[Dict[str, Union[List[Tensor], Tensor]]] = None

    def add_state(
            self,
            name: str,
            default: Union[list, Tensor],
            dist_reduce_fx: Optional[Union[str, Callable]] = None,
            persistent: bool = False,
    ) -> None:
        if not isinstance(default, (Tensor, list)) or (isinstance(default, list) and default):
            raise ValueError("state variable must be a tensor or any empty list (where you can append tensors)")

        if dist_reduce_fx == "sum":
            dist_reduce_fx = dim_zero_sum
        elif dist_reduce_fx == "mean":
            dist_reduce_fx = dim_zero_mean
        elif dist_reduce_fx == "max":
            dist_reduce_fx = dim_zero_max
        elif dist_reduce_fx == "min":
            dist_reduce_fx = dim_zero_min
        elif dist_reduce_fx == "cat":
            dist_reduce_fx = dim_zero_cat
        elif dist_reduce_fx is not None and not callable(dist_reduce_fx):
            raise ValueError("`dist_reduce_fx` must be callable or one of ['mean', 'sum', 'cat', None]")

        if isinstance(default, Tensor):
            default = default.contiguous()

        setattr(self, name, default)

        self._defaults[name] = deepcopy(default)
        self._persistent[name] = persistent
        self._reductions[name] = dist_reduce_fx

    @torch.jit.unused
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Automatically calls ``update()``.

        Returns the metric value over inputs if ``compute_on_step`` is True.
        """
        # add current step
        if self._is_synced:
            raise TorchMetricsUserError(
                "The Metric shouldn't be synced when performing ``update``. "
                "HINT: Did you forget to call ``unsync`` ?."
            )

        with torch.no_grad():
            self.update(*args, **kwargs)

        if self.compute_on_step:
            self._to_sync = self.dist_sync_on_step
            # skip restore cache operation from compute as cache is stored below.
            self._should_unsync = False

            # save context before switch
            cache = {attr: getattr(self, attr) for attr in self._defaults}

            # call reset, update, compute, on single batch
            self.reset()
            self.update(*args, **kwargs)
            self._forward_cache = self.compute()

            # restore context
            for attr, val in cache.items():
                setattr(self, attr, val)
            self._is_synced = False

            self._should_unsync = True
            self._to_sync = True
            self._computed = None

            return self._forward_cache

    def _sync_dist(self, dist_sync_fn: Callable = gather_all_tensors, process_group: Optional[Any] = None) -> None:
        input_dict = {attr: getattr(self, attr) for attr in self._reductions}

        for attr, reduction_fn in self._reductions.items():
            # pre-concatenate metric states that are lists to reduce number of all_gather operations
            if reduction_fn == dim_zero_cat and isinstance(input_dict[attr], list) and len(input_dict[attr]) > 1:
                input_dict[attr] = [dim_zero_cat(input_dict[attr])]

        output_dict = apply_to_collection(
            input_dict,
            Tensor,
            dist_sync_fn,
            group=process_group or self.process_group,
        )

        for attr, reduction_fn in self._reductions.items():
            # pre-processing ops (stack or flatten for inputs)
            if isinstance(output_dict[attr][0], Tensor):
                output_dict[attr] = torch.stack(output_dict[attr])
            elif isinstance(output_dict[attr][0], list):
                output_dict[attr] = _flatten(output_dict[attr])

            if not (callable(reduction_fn) or reduction_fn is None):
                raise TypeError("reduction_fn must be callable or None")
            reduced = reduction_fn(output_dict[attr]) if reduction_fn is not None else output_dict[attr]
            setattr(self, attr, reduced)

    def _wrap_update(self, update: Callable) -> Callable:
        @functools.wraps(update)
        def wrapped_func(*args: Any, **kwargs: Any) -> Optional[Any]:
            self._computed = None
            self._update_called = True
            return update(*args, **kwargs)

        return wrapped_func

    def sync(
            self,
            dist_sync_fn: Optional[Callable] = None,
            process_group: Optional[Any] = None,
            should_sync: bool = True,
            distributed_available: Optional[Callable] = jit_distributed_available,
    ) -> None:
        """Sync function for manually controlling when metrics states should be synced across processes.

        Args:
            dist_sync_fn: Function to be used to perform states synchronization
            process_group:
                Specify the process group on which synchronization is called.
                default: None (which selects the entire world)
            should_sync: Whether to apply to state synchronization. This will have an impact
                only when running in a distributed setting.
            distributed_available: Function to determine if we are running inside a distributed setting
        """
        if self._is_synced and should_sync:
            raise TorchMetricsUserError("The Metric has already been synced.")

        is_distributed = distributed_available() if callable(distributed_available) else None

        if not should_sync or not is_distributed:
            return

        if dist_sync_fn is None:
            dist_sync_fn = gather_all_tensors

        # cache prior to syncing
        self._cache = {attr: getattr(self, attr) for attr in self._defaults}

        # sync
        self._sync_dist(dist_sync_fn, process_group=process_group)
        self._is_synced = True

    def unsync(self, should_unsync: bool = True) -> None:
        """Unsync function for manually controlling when metrics states should be reverted back to their local
        states.

        Args:
            should_unsync: Whether to perform unsync
        """
        if not should_unsync:
            return

        if not self._is_synced:
            raise TorchMetricsUserError("The Metric has already been un-synced.")

        if self._cache is None:
            raise TorchMetricsUserError("The internal cache should exist to unsync the Metric.")

        # if we synced, restore to cache so that we can continue to accumulate un-synced state
        for attr, val in self._cache.items():
            setattr(self, attr, val)
        self._is_synced = False
        self._cache = None

    @contextmanager
    def sync_context(
            self,
            dist_sync_fn: Optional[Callable] = None,
            process_group: Optional[Any] = None,
            should_sync: bool = True,
            should_unsync: bool = True,
            distributed_available: Optional[Callable] = jit_distributed_available,
    ) -> Generator:
        """Context manager to synchronize the states between processes when running in a distributed setting and
        restore the local cache states after yielding.

        Args:
            dist_sync_fn: Function to be used to perform states synchronization
            process_group:
                Specify the process group on which synchronization is called.
                default: None (which selects the entire world)
            should_sync: Whether to apply to state synchronization. This will have an impact
                only when running in a distributed setting.
            should_unsync: Whether to restore the cache state so that the metrics can
                continue to be accumulated.
            distributed_available: Function to determine if we are running inside a distributed setting
        """
        self.sync(
            dist_sync_fn=dist_sync_fn,
            process_group=process_group,
            should_sync=should_sync,
            distributed_available=distributed_available,
        )

        yield

        self.unsync(should_unsync=self._is_synced and should_unsync)

    def _wrap_compute(self, compute: Callable) -> Callable:
        @functools.wraps(compute)
        def wrapped_func(*args: Any, **kwargs: Any) -> Any:
            # return cached value
            if self._computed is not None:
                return self._computed

            # compute relies on the sync context manager to gather the states across processes and apply reduction
            # if synchronization happened, the current rank accumulated states will be restored to keep
            # accumulation going if ``should_unsync=True``,
            with self.sync_context(
                    dist_sync_fn=self.dist_sync_fn, should_sync=self._to_sync, should_unsync=self._should_unsync
            ):
                self._computed = compute(*args, **kwargs)

            return self._computed

        return wrapped_func

    @abstractmethod
    def update(self, *_: Any, **__: Any) -> None:
        """Override this method to update the state variables of your metric class."""

    @abstractmethod
    def compute(self) -> Any:
        """Override this method to compute the final metric value from state variables synchronized across the
        distributed backend."""

    def reset(self) -> None:
        """This method automatically resets the metric state variables to their default value."""
        self._update_called = False
        self._forward_cache = None
        # lower lightning versions requires this implicitly to log metric objects correctly in self.log
        self._computed = None

        for attr, default in self._defaults.items():
            current_val = getattr(self, attr)
            if isinstance(default, Tensor):
                setattr(self, attr, default.detach().clone().to(current_val.device))
            else:
                setattr(self, attr, [])

        # reset internal states
        self._cache = None
        self._is_synced = False

    def clone(self) -> "_Metric":
        """Make a copy of the metric."""
        return deepcopy(self)

    def __getstate__(self) -> Dict[str, Any]:
        # ignore update and compute functions for pickling
        return {k: v for k, v in self.__dict__.items() if k not in ["update", "compute", "_update_signature"]}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        # manually restore update and compute functions for pickling
        self.__dict__.update(state)
        self._update_signature = inspect.signature(self.update)
        self.update: Callable = self._wrap_update(self.update)  # type: ignore
        self.compute: Callable = self._wrap_compute(self.compute)  # type: ignore

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("higher_is_better", "is_differentiable"):
            raise RuntimeError(f"Can't change const `{name}`.")
        super().__setattr__(name, value)

    @property
    def device(self) -> "torch.device":
        """Return the device of the metric."""
        return self._device

    def type(self, dst_type: Union[str, torch.dtype]) -> "_Metric":
        """Method override default and prevent dtype casting.

        Please use `metric.set_dtype(dtype)` instead.
        """
        return self

    def float(self) -> "_Metric":
        """Method override default and prevent dtype casting.

        Please use `metric.set_dtype(dtype)` instead.
        """
        return self

    def double(self) -> "_Metric":
        """Method override default and prevent dtype casting.

        Please use `metric.set_dtype(dtype)` instead.
        """
        return self

    def half(self) -> "_Metric":
        """Method override default and prevent dtype casting.

        Please use `metric.set_dtype(dtype)` instead.
        """
        return self

    def set_dtype(self, dst_type: Union[str, torch.dtype]) -> None:
        """Special version of `type` for transferring all metric states to specific dtype
        Arguments:
            dst_type (type or string): the desired type
        """
        return super().type(dst_type)

    def _apply(self, fn: Callable) -> nn.Module:
        """Overwrite _apply function such that we can also move metric states to the correct device when `.to`,
        `.cuda`, etc methods are called."""
        this = super()._apply(fn)
        # Also apply fn to metric states and defaults
        for key, value in this._defaults.items():
            if isinstance(value, Tensor):
                this._defaults[key] = fn(value)
            elif isinstance(value, Sequence):
                this._defaults[key] = [fn(v) for v in value]

            current_val = getattr(this, key)
            if isinstance(current_val, Tensor):
                setattr(this, key, fn(current_val))
            elif isinstance(current_val, Sequence):
                setattr(this, key, [fn(cur_v) for cur_v in current_val])
            else:
                raise TypeError(
                    "Expected metric state to be either a Tensor" f"or a list of Tensor, but encountered {current_val}"
                )

        # make sure to update the device attribute
        # if the dummy tensor moves device by fn function we should also update the attribute
        self._device = fn(torch.zeros(1, device=self.device)).device

        # Additional apply to forward cache and computed attributes (may be nested)
        if this._computed is not None:
            this._computed = apply_to_collection(this._computed, Tensor, fn)
        if this._forward_cache is not None:
            this._forward_cache = apply_to_collection(this._forward_cache, Tensor, fn)

        return this

    def persistent(self, mode: bool = False) -> None:
        """Method for post-init to change if metric states should be saved to its state_dict."""
        for key in self._persistent:
            self._persistent[key] = mode

    def state_dict(
            self,
            destination: Dict[str, Any] = None,
            prefix: str = "",
            keep_vars: bool = False,
    ) -> Optional[Dict[str, Any]]:
        destination = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        # Register metric states to be part of the state_dict
        for key in self._defaults:
            if not self._persistent[key]:
                continue
            current_val = getattr(self, key)
            if not keep_vars:
                if isinstance(current_val, Tensor):
                    current_val = current_val.detach()
                elif isinstance(current_val, list):
                    current_val = [cur_v.detach() if isinstance(cur_v, Tensor) else cur_v for cur_v in current_val]
            destination[prefix + key] = deepcopy(current_val)  # type: ignore
        return destination

    def _load_from_state_dict(
            self,
            state_dict: dict,
            prefix: str,
            local_metadata: dict,
            strict: bool,
            missing_keys: List[str],
            unexpected_keys: List[str],
            error_msgs: List[str],
    ) -> None:
        """Loads metric states from state_dict."""

        for key in self._defaults:
            name = prefix + key
            if name in state_dict:
                setattr(self, key, state_dict.pop(name))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
        )

    def _filter_kwargs(self, **kwargs: Any) -> Dict[str, Any]:
        """filter kwargs such that they match the update signature of the metric."""

        # filter all parameters based on update signature except those of
        # type VAR_POSITIONAL (*args) and VAR_KEYWORD (**kwargs)
        _params = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        _sign_params = self._update_signature.parameters
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if (k in _sign_params.keys() and _sign_params[k].kind not in _params)
        }

        # if no kwargs filtered, return al kwargs as default
        if not filtered_kwargs:
            filtered_kwargs = kwargs
        return filtered_kwargs

    def __hash__(self) -> int:
        # we need to add the id here, since PyTorch requires a module hash to be unique.
        # Internally, PyTorch nn.Module relies on that for children discovery
        # (see https://github.com/pytorch/pytorch/blob/v1.9.0/torch/nn/modules/module.py#L1544)
        # For metrics that include tensors it is not a problem,
        # since their hash is unique based on the memory location but we cannot rely on that for every metric.
        hash_vals = [self.__class__.__name__, id(self)]

        for key in self._defaults:
            val = getattr(self, key)
            # Special case: allow list values, so long
            # as their elements are hashable
            if hasattr(val, "__iter__") and not isinstance(val, Tensor):
                hash_vals.extend(val)
            else:
                hash_vals.append(val)

        return hash(tuple(hash_vals))

    def __add__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.add, self, other)

    def __and__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.bitwise_and, self, other)

    # Fixme: this shall return bool instead of Metric
    def __eq__(self, other: "Metric") -> "Metric":  # type: ignore
        return CompositionalMetric(torch.eq, self, other)

    def __floordiv__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.floor_divide, self, other)

    def __ge__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.ge, self, other)

    def __gt__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.gt, self, other)

    def __le__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.le, self, other)

    def __lt__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.lt, self, other)

    def __matmul__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.matmul, self, other)

    def __mod__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.fmod, self, other)

    def __mul__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.mul, self, other)

    # Fixme: this shall return bool instead of Metric
    def __ne__(self, other: "Metric") -> "Metric":  # type: ignore
        return CompositionalMetric(torch.ne, self, other)

    def __or__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.bitwise_or, self, other)

    def __pow__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.pow, self, other)

    def __radd__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.add, other, self)

    def __rand__(self, other: "Metric") -> "Metric":
        # swap them since bitwise_and only supports that way and it's commutative
        return CompositionalMetric(torch.bitwise_and, self, other)

    def __rfloordiv__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.floor_divide, other, self)

    def __rmatmul__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.matmul, other, self)

    def __rmod__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.fmod, other, self)

    def __rmul__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.mul, other, self)

    def __ror__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.bitwise_or, other, self)

    def __rpow__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.pow, other, self)

    def __rsub__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.sub, other, self)

    def __rtruediv__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.true_divide, other, self)

    def __rxor__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.bitwise_xor, other, self)

    def __sub__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.sub, self, other)

    def __truediv__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.true_divide, self, other)

    def __xor__(self, other: "Metric") -> "Metric":
        return CompositionalMetric(torch.bitwise_xor, self, other)

    def __abs__(self) -> "Metric":
        return CompositionalMetric(torch.abs, self, None)

    def __inv__(self) -> "Metric":
        return CompositionalMetric(torch.bitwise_not, self, None)

    def __invert__(self) -> "Metric":
        return self.__inv__()

    def __neg__(self) -> "Metric":
        return CompositionalMetric(_neg, self, None)

    def __pos__(self) -> "Metric":
        return CompositionalMetric(torch.abs, self, None)

    def __getitem__(self, idx: int) -> "Metric":
        return CompositionalMetric(lambda x: x[idx], self, None)



class CompositionalMetric(_Metric):
    """Composition of two metrics with a specific operator which will be executed upon metrics compute."""

    def __init__(
            self,
            operator: Callable,
            metric_a: Union[_Metric, int, float, Tensor],
            metric_b: Union[_Metric, int, float, Tensor, None],
    ) -> None:
        """
        Args:
            operator: the operator taking in one (if metric_b is None)
                or two arguments. Will be applied to outputs of metric_a.compute()
                and (optionally if metric_b is not None) metric_b.compute()
            metric_a: first metric whose compute() result is the first argument of operator
            metric_b: second metric whose compute() result is the second argument of operator.
                For operators taking in only one input, this should be None
        """
        super().__init__()

        self.op = operator

        if isinstance(metric_a, Tensor):
            self.register_buffer("metric_a", metric_a)
        else:
            self.metric_a = metric_a

        if isinstance(metric_b, Tensor):
            self.register_buffer("metric_b", metric_b)
        else:
            self.metric_b = metric_b

    def _sync_dist(self, dist_sync_fn: Optional[Callable] = None, process_group: Optional[Any] = None) -> None:
        # No syncing required here. syncing will be done in metric_a and metric_b
        pass

    def update(self, *args: Any, **kwargs: Any) -> None:
        if isinstance(self.metric_a, Metric):
            self.metric_a.update(*args, **self.metric_a._filter_kwargs(**kwargs))

        if isinstance(self.metric_b, Metric):
            self.metric_b.update(*args, **self.metric_b._filter_kwargs(**kwargs))

    def compute(self) -> Any:

        # also some parsing for kwargs?
        if isinstance(self.metric_a, Metric):
            val_a = self.metric_a.compute()
        else:
            val_a = self.metric_a

        if isinstance(self.metric_b, Metric):
            val_b = self.metric_b.compute()
        else:
            val_b = self.metric_b

        if val_b is None:
            return self.op(val_a)

        return self.op(val_a, val_b)

    def reset(self) -> None:
        if isinstance(self.metric_a, Metric):
            self.metric_a.reset()

        if isinstance(self.metric_b, Metric):
            self.metric_b.reset()

    def persistent(self, mode: bool = False) -> None:
        if isinstance(self.metric_a, Metric):
            self.metric_a.persistent(mode=mode)
        if isinstance(self.metric_b, Metric):
            self.metric_b.persistent(mode=mode)

    def __repr__(self) -> str:
        _op_metrics = f"(\n  {self.op.__name__}(\n    {repr(self.metric_a)},\n    {repr(self.metric_b)}\n  )\n)"
        repr_str = self.__class__.__name__ + _op_metrics

        return repr_str


class MetricCollection_(nn.ModuleDict):
    def __init__(
            self,
            metrics: Union[_Metric, Sequence[_Metric], Dict[str, _Metric]],
            *additional_metrics: _Metric,
            prefix: Optional[str] = None,
            postfix: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.add_metrics(metrics, *additional_metrics)

        self.prefix = self._check_arg(prefix, "prefix")
        self.postfix = self._check_arg(postfix, "postfix")

    @torch.jit.unused
    def forward(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Iteratively call forward for each metric.

        Positional arguments (args) will be passed to every metric in the collection, while keyword arguments (kwargs)
        will be filtered based on the signature of the individual metric.
        """
        return {k: m(*args, **m._filter_kwargs(**kwargs)) for k, m in self.items()}

    def update(self, *args: Any, **kwargs: Any) -> None:
        """Iteratively call update for each metric.

        Positional arguments (args) will be passed to every metric in the collection, while keyword arguments (kwargs)
        will be filtered based on the signature of the individual metric.
        """
        for _, m in self.items(keep_base=True):
            m_kwargs = m._filter_kwargs(**kwargs)
            m.update(*args, **m_kwargs)

    def compute(self) -> Dict[str, Any]:
        return {k: m.compute() for k, m in self.items()}

    def reset(self) -> None:
        """Iteratively call reset for each metric."""
        for _, m in self.items(keep_base=True):
            m.reset()

    def clone(self, prefix: Optional[str] = None, postfix: Optional[str] = None) -> "MetricCollection_":
        """Make a copy of the metric collection
        Args:
            prefix: a string to append in front of the metric keys
            postfix: a string to append after the keys of the output dict

        """
        mc = deepcopy(self)
        if prefix:
            mc.prefix = self._check_arg(prefix, "prefix")
        if postfix:
            mc.postfix = self._check_arg(postfix, "postfix")
        return mc

    def persistent(self, mode: bool = True) -> None:
        """Method for post-init to change if metric states should be saved to its state_dict."""
        for _, m in self.items(keep_base=True):
            m.persistent(mode)

    def add_metrics(
            self, metrics: Union[_Metric, Sequence[_Metric], Dict[str, _Metric]], *additional_metrics: _Metric
    ) -> None:
        """Add new metrics to Metric Collection."""
        if isinstance(metrics, Metric):
            # set compatible with original type expectations
            metrics = [metrics]
        if isinstance(metrics, Sequence):
            # prepare for optional additions
            metrics = list(metrics)
            remain: list = []
            for m in additional_metrics:
                (metrics if isinstance(m, Metric) else remain).append(m)

        elif additional_metrics:
            raise ValueError(
                f"You have passes extra arguments {additional_metrics} which are not compatible"
                f" with first passed dictionary {metrics} so they will be ignored."
            )

        if isinstance(metrics, dict):
            # Check all values are metrics
            # Make sure that metrics are added in deterministic order
            for name in sorted(metrics.keys()):
                metric = metrics[name]
                if not isinstance(metric, Metric):
                    raise ValueError(
                        f"Value {metric} belonging to key {name} is not an instance of `pl.metrics.Metric`"
                    )
                self[name] = metric
        elif isinstance(metrics, Sequence):
            for metric in metrics:
                if not isinstance(metric, Metric):
                    raise ValueError(f"Input {metric} to `MetricCollection` is not a instance of `pl.metrics.Metric`")
                name = metric.__class__.__name__
                if name in self:
                    raise ValueError(f"Encountered two metrics both named {name}")
                self[name] = metric
        else:
            raise ValueError("Unknown input to MetricCollection.")

    def _set_name(self, base: str) -> str:
        name = base if self.prefix is None else self.prefix + base
        name = name if self.postfix is None else name + self.postfix
        return name

    def _to_renamed_ordered_dict(self) -> OrderedDict:
        od = OrderedDict()
        for k, v in self._modules.items():
            od[self._set_name(k)] = v
        return od

    def keys(self, keep_base: bool = False) -> Iterable[Hashable]:
        r"""Return an iterable of the ModuleDict key.
        Args:
            keep_base: Whether to add prefix/postfix on the items collection.
        """
        if keep_base:
            return self._modules.keys()
        return self._to_renamed_ordered_dict().keys()

    def items(self, keep_base: bool = False) -> Iterable[Tuple[str, nn.Module]]:
        r"""Return an iterable of the ModuleDict key/value pairs.
        Args:
            keep_base: Whether to add prefix/postfix on the items collection.
        """
        if keep_base:
            return self._modules.items()
        return self._to_renamed_ordered_dict().items()

    @staticmethod
    def _check_arg(arg: Optional[str], name: str) -> Optional[str]:
        if arg is None or isinstance(arg, str):
            return arg
        raise ValueError(f"Expected input `{name}` to be a string, but got {type(arg)}")

    def __repr__(self) -> str:
        repr_str = super().__repr__()[:-2]
        if self.prefix:
            repr_str += f",\n  prefix={self.prefix}{',' if self.postfix else ''}"
        if self.postfix:
            repr_str += f"{',' if not self.prefix else ''}\n  postfix={self.postfix}"
        return repr_str + "\n)"


class Metric(_Metric):
    r"""
    This implementation refers to :class:`~torchmetrics.Metric`.

    .. warning:: This metric is deprecated, use ``torchmetrics.Metric``. Will be removed in v1.5.0.
    """

    def __init__(
            self,
            compute_on_step: bool = True,
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
            dist_sync_fn: Callable = None,
    ):
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

    def __hash__(self):
        return super().__hash__()

    def __add__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.add, self, other)

    def __and__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.bitwise_and, self, other)

    def __eq__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.eq, self, other)

    def __floordiv__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.floor_divide, self, other)

    def __ge__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.ge, self, other)

    def __gt__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.gt, self, other)

    def __le__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.le, self, other)

    def __lt__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.lt, self, other)

    def __matmul__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.matmul, self, other)

    def __mod__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.fmod, self, other)

    def __mul__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.mul, self, other)

    def __ne__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.ne, self, other)

    def __or__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.bitwise_or, self, other)

    def __pow__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.pow, self, other)

    def __radd__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.add, other, self)

    def __rand__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric

        # swap them since bitwise_and only supports that way and it's commutative
        return CompositionalMetric(torch.bitwise_and, self, other)

    def __rfloordiv__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.floor_divide, other, self)

    def __rmatmul__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.matmul, other, self)

    def __rmod__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.fmod, other, self)

    def __rmul__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.mul, other, self)

    def __ror__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.bitwise_or, other, self)

    def __rpow__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.pow, other, self)

    def __rsub__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.sub, other, self)

    def __rtruediv__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.true_divide, other, self)

    def __rxor__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.bitwise_xor, other, self)

    def __sub__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.sub, self, other)

    def __truediv__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.true_divide, self, other)

    def __xor__(self, other: Any):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.bitwise_xor, self, other)

    def __abs__(self):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.abs, self, None)

    def __inv__(self):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.bitwise_not, self, None)

    def __invert__(self):
        return self.__inv__()

    def __neg__(self):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(_neg, self, None)

    def __pos__(self):
        from pytorch_lightning.metrics.compositional import CompositionalMetric
        return CompositionalMetric(torch.abs, self, None)


def _neg(tensor: torch.Tensor):
    return -torch.abs(tensor)


class MicroF1(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("prec_d", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("rec_d", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, p, g):

        self.n += len(g.intersection(p))
        self.prec_d += len(p)
        self.rec_d += len(g)

    def compute(self):
        p = self.n.float() / self.prec_d
        r = self.n.float() / self.rec_d
        return (2 * p * r / (p + r)) if (p + r) > 0 else (p + r)


class MacroF1(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("n", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("d", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, p, g):

        prec = len(g.intersection(p)) / len(p)
        rec = len(g.intersection(p)) / len(g) if g else 0.0

        self.n += (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else (prec + rec)
        self.d += 1

    def compute(self):
        return (self.n / self.d) if self.d > 0 else self.d


class MicroPrecision(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("d", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, p, g):
        self.n += len(g.intersection(p))
        self.d += len(p)

    def compute(self):
        return (self.n.float() / self.d) if self.d > 0 else self.d


class MacroPrecision(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("n", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("d", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, p, g):
        self.n += len(g.intersection(p)) / len(p)
        self.d += 1

    def compute(self):
        return (self.n / self.d) if self.d > 0 else self.d


class MicroRecall(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("d", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, p, g):
        self.n += len(g.intersection(p))
        self.d += len(g)

    def compute(self):
        return (self.n.float() / self.d) if self.d > 0 else self.d


class MacroRecall(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("n", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("d", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, p, g):
        self.n += len(g.intersection(p)) / len(g) if g else 0.0
        self.d += 1

    def compute(self):
        return (self.n / self.d) if self.d > 0 else self.d

# The following two classes are not inherited from https://github.com/nicola-decao/efficient-autoregressive-EL
#  and are implemented in this project.


class _EvaluationScores:
    def __init__(self, is_micro):
        self.is_micro = is_micro
        if is_micro:
            self.f1 = MicroF1()
            self.p = MicroPrecision()
            self.r = MicroRecall()
        else:
            self.f1 = MacroF1()
            self.p = MacroPrecision()
            self.r = MacroRecall()

    def record_results(self, prediction, gold):
        self.f1(prediction, gold)
        self.p(prediction, gold)
        self.r(prediction, gold)

    def __str__(self):
        im = "Micro" if self.is_micro else "Macro"
        return f"\t{im} evaluation results: F1: {self.f1.compute() * 100:.3f}%\tP: {self.p.compute() * 100:.3f}%" \
               f"\t R: {self.r.compute() * 100:.3f}%"


class EntityEvaluationScores:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.micro_mention_detection = _EvaluationScores(True)
        self.macro_mention_detection = _EvaluationScores(False)
        self.micro_entity_linking = _EvaluationScores(True)
        self.macro_entity_linking = _EvaluationScores(False)

    def record_mention_detection_results(self, prediction, gold):
        self.micro_mention_detection.record_results(prediction, gold)
        self.macro_mention_detection.record_results(prediction, gold)

    def record_entity_linking_results(self, prediction, gold):
        self.micro_entity_linking.record_results(prediction, gold)
        self.macro_entity_linking.record_results(prediction, gold)

    def __str__(self):
        return f"Evaluated model for set: {self.dataset_name} (Entity Linking)\n" \
               f"{str(self.macro_entity_linking)}\n" \
               f"{str(self.micro_entity_linking)}\n" \
               f"Evaluated model for set: {self.dataset_name} (Mention Detection)\n" \
               f"{str(self.macro_mention_detection)}\n" \
               f"{str(self.micro_mention_detection)}"


class InOutMentionEvaluationResult:
    def __init__(self, activation_threshold=0.5, vocab_index_of_o=-1):
        self.activation_threshold = activation_threshold
        self.vocab_index_of_o = vocab_index_of_o
        self.total_predictions = 0.0
        self.correct_predictions = 0.0
        self.total_true_predictions = 0.0
        self.correct_true_predictions = 0.0
        self.total_false_predictions = 0.0
        self.correct_false_predictions = 0.0

    def _preprocess_logits(self, subword_logits):
        if self.vocab_index_of_o > -1:
            return (subword_logits.argmax(-1) != self.vocab_index_of_o).bool()
        else:
            return (subword_logits > self.activation_threshold).squeeze(-1)

    def update_scores(self, inputs_eval_mask, s_mentions_is_in_mention, subword_logits):
        self.total_predictions += inputs_eval_mask.sum().item()
        for em, ac, pr in zip(inputs_eval_mask, s_mentions_is_in_mention.bool(),
                              self._preprocess_logits(subword_logits)):
            for m, a, p in zip(em, ac, pr):
                if m:
                    if a == p:
                        self.correct_predictions += 1.0
                    if a:
                        self.total_true_predictions += 1.0
                        if p:
                            self.correct_true_predictions += 1.0
                    else:
                        self.total_false_predictions += 1.0
                        if not p:
                            self.correct_false_predictions += 1.0

    @property
    def overall_mention_detection_accuracy(self):
        return self.correct_predictions * 100 / self.total_predictions if self.total_predictions > 0.0 else 0.0

    @property
    def in_mention_mention_detection_accuracy(self):
        return self.correct_true_predictions * 100 / self.total_true_predictions \
            if self.total_true_predictions > 0.0 else 0.0

    @property
    def out_of_mention_overall_mention_detection_accuracy(self):
        return self.correct_false_predictions * 100 / self.total_false_predictions \
            if self.total_false_predictions > 0.0 else 0.0

    def __str__(self):
        return f"Subword-level mention detection accuracy = {self.overall_mention_detection_accuracy:.3f}% " \
               f"({int(self.correct_predictions)}/{int(self.total_predictions)})\n" \
               f"\t    In-Mention Subword-level mention detection accuracy = " \
               f"{self.in_mention_mention_detection_accuracy:.3f}% " \
               f"({int(self.correct_true_predictions)}/{int(self.total_true_predictions)})\n" \
               f"\tOut-of-Mention Subword-level mention detection accuracy = " \
               f"{self.out_of_mention_overall_mention_detection_accuracy:.3f}% " \
               f"({int(self.correct_false_predictions)}/{int(self.total_false_predictions)})"
