from typing import List

from .. import util
from ..util import slurp, spit, tree

_DEFAULT_IGNORE = (
    r"(^|[\\/])__pycache__([\\/]|$)"
    r"|(^|[\\/])node_modules([\\/]|$)"
    r"|(^|[\\/])bower_components([\\/]|$)"
    r"|(^|[\\/])jspm_packages([\\/]|$)"
    r"|(^|[\\/])(\.git|\.hg|\.svn)([\\/]|$)"
    r"|\.(?:gitignore)$"
    r"|(^|[\\/])(vendor|third_party|third-party|external|extern|deps|dep|subprojects)([\\/]|$)"
    r"|(^|[\\/])(env-[^\\/]+|venv-[^\\/]+|venv)([\\/]|$)"
    r"|(^|[\\/])[^\\/]+\.egg-info([\\/]|$)"
    r"|(^|[\\/])(\.hypothesis|\.pytest_cache|\.mypy_cache|\.ruff_cache)([\\/]|$)"
    r"|(^|[\\/])(\.deps|\.libs|autom4te\.cache|CMakeFiles|cmake-build-[^\\/]+|_deps)([\\/]|$)"
    r"|(^|[\\/])[^\\/]+\.dSYM([\\/]|$)"
    r"|(^|[\\/])(build|dist|out|coverage|target)([\\/]|$)"
    r"|^\..*"
    r"|~$|\.pyc$|\.pyo$|\.class$|Thumbs\.db$|\.DS_Store$|\.log$|\.bak$|\.swp$|\.swo$|\.tmp$|\.temp$|\.lock$"
    r"|\.(?:o|obj|lo|la|a|so(?:\.\d+)*|dylib|dll|exe|pdb|ilk|idb|gcda|gcno|gcov)$"
    r"|CMakeCache\.txt$|cmake_install\.cmake$|compile_commands\.json$"
    r"|config\.(?:log|status|cache)$|aclocal\.m4$|ltmain\.sh$|libtool$|stamp-h1$|test-suite\.log$"
    r"|package-lock\.json$|npm-shrinkwrap\.json$|pnpm-lock\.yaml$|yarn\.lock$|Cargo\.lock$|\.tsbuildinfo$"
)


def code_ls(path: str) -> List[str]:
    return list(util.deep_ls(path, ignore=_DEFAULT_IGNORE))
