from typing import List

from .. import util
from ..util import slurp, spit, tree

_DEFAULT_IGNORE = r"(^__pycache__|^node_modules|\.git/|/env-.*|\.egg-info/.*|^venv|^\..*|~$|\.pyc$|Thumbs\.db$|^build[\\/]|^dist[\\/]|^coverage[\\/]|\.log$|\.lock$|\.bak$|\.swp$|\.swo$|\.tmp$|\.temp$|\.class$|^target$|^Cargo\.lock$)"


def code_ls(path: str) -> List[str]:
    return list(util.deep_ls(path, ignore=_DEFAULT_IGNORE))
