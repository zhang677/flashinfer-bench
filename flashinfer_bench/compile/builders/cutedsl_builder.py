"""Builder for CuTeDSL GPU kernels."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable, ClassVar

from flashinfer_bench.compile.builder import Builder
from flashinfer_bench.compile.runnable import Runnable
from flashinfer_bench.data import Definition, Solution, SupportedLanguages

from .python_builder import PythonBuilder

_original_generate_mlir = None


def _generate_mlir_cached(
    self,
    funcBody,
    kwargs,
    function_name,
    gpu_module_attrs,
    args,
    args_spec,
    pipeline,
    no_cache,
    no_jit_engine,
    compile_only,
    location=None,
):
    return _original_generate_mlir(
        self,
        funcBody,
        kwargs,
        function_name,
        gpu_module_attrs,
        args,
        args_spec,
        pipeline,
        False,
        no_jit_engine,
        compile_only,
        location=location,
    )


def patch_cute_compile_cache() -> None:
    """Patch BaseDSL.generate_mlir to force no_cache=False, enabling MLIR caching."""
    global _original_generate_mlir
    from cutlass.base_dsl.dsl import BaseDSL

    _original_generate_mlir = BaseDSL.generate_mlir
    BaseDSL.generate_mlir = _generate_mlir_cached


def unpatch_cute_compile_cache() -> None:
    """Restore the original BaseDSL.generate_mlir method."""
    global _original_generate_mlir
    if _original_generate_mlir is not None:
        from cutlass.base_dsl.dsl import BaseDSL

        BaseDSL.generate_mlir = _original_generate_mlir
        _original_generate_mlir = None


class CuteDSLBuilder(PythonBuilder):
    """Builder for CuTeDSL solutions.

    This builder extends PythonBuilder to handle CuTeDSL GPU kernels. CuTeDSL code
    is Python-based, so the build process is similar to PythonBuilder. Before building,
    it patches BaseDSL.generate_mlir to enable MLIR compilation caching, and restores
    the original method on cleanup.
    """

    _PACKAGE_PREFIX: ClassVar[str] = "fib_cutedsl_"
    """Prefix for cache keys to distinguish CuTeDSL solutions from pure Python ones."""

    _BUILD_DIR_NAME: ClassVar[str] = "cutedsl"
    """Subdirectory under FIB_CACHE_PATH where build results are stored."""

    def __init__(self) -> None:
        Builder.__init__(self, self._PACKAGE_PREFIX, self._BUILD_DIR_NAME)

    @staticmethod
    def is_available() -> bool:
        """Check if CuTeDSL (CUTLASS) is available in the current environment.

        Returns
        -------
        bool
            True if the cutlass package is installed, False otherwise.
        """
        return importlib.util.find_spec("cutlass") is not None

    def can_build(self, solution: Solution) -> bool:
        """Check if this builder can build the given solution.
        The solution should be CuTeDSL source code.

        Parameters
        ----------
        solution : Solution
            Solution to check

        Returns
        -------
        bool
            True if solution language is CuTeDSL
        """
        return solution.spec.language == SupportedLanguages.CUTEDSL

    def _get_cleaner(self, package: str, build_path: Path) -> Callable[[], None]:
        """Create a cleaner that also unpatches CuTeDSL compile cache.

        Parameters
        ----------
        package : str
            The package name to unload from sys.modules.
        build_path : Path
            The directory to delete.

        Returns
        -------
        Callable[[], None]
            A function that performs the cleanup.
        """
        base_cleaner = super()._get_cleaner(package, build_path)

        def cleaner() -> None:
            unpatch_cute_compile_cache()
            base_cleaner()

        return cleaner

    def build(self, definition: Definition, solution: Solution) -> Runnable:
        """Build a CuTeDSL solution into a runnable.

        Patches BaseDSL.generate_mlir to enable MLIR caching before delegating
        to PythonBuilder.build(). The patch is removed on cleanup.

        Parameters
        ----------
        definition : Definition
            The problem definition.
        solution : Solution
            The CuTeDSL solution to build.

        Returns
        -------
        Runnable
            An executable wrapper around the CuTeDSL kernel.
        """
        patch_cute_compile_cache()
        result = super().build(definition, solution)
        result.metadata.build_type = "cutedsl"
        return result
