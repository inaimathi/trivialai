import os

from . import util

logger = util.getLogger("trivialai.filesystem")


class FilesystemMixin:
    def edit_file(
        self,
        file_path,
        system,
        prompt,
        after_save=None,
        consider_current=True,
        retries=5,
    ):
        full_system = "\n".join(
            [
                system,
                f"The current contents of {file_path} is {util.slurp(file_path)}"
                if (os.path.isfile(file_path) and consider_current)
                else f"The file {file_path} currently doesn't exist.",
                f"What changes would you make to the file {file_path}? Return only the new contents of {file_path} and no other information.",
            ]
        )
        cont = self.generate(full_system, prompt).content
        util.spit(file_path, util.strip_md_code(cont))
        if after_save is not None:
            after_save(file_path)

    def relevant_files(
        self, in_dir, prompt, ignore=None, focus=None, must_exist=None, retries=5
    ):
        return _relevant_files(
            self,
            in_dir,
            prompt,
            ignore=ignore,
            focus=focus,
            must_exist=must_exist,
            retries=retries,
        )

    def target_files(
        self,
        in_dir,
        prompt,
        system=None,
        ignore=None,
        focus=None,
        must_exist=False,
        retries=5,
    ):
        return _target_files(
            self,
            in_dir,
            prompt,
            system=system,
            ignore=ignore,
            focus=focus,
            must_exist=must_exist,
            retries=retries,
        )

    def edit_files_considering(
        self, consider_paths, edit_paths, prompt, after_save=None, retries=5
    ):
        return _edit_files_considering(
            self, consider_paths, edit_paths, prompt, after_save, retries
        )
        pass

    def edit_directory(
        self,
        in_dir,
        prompt,
        after_save=None,
        out_dir=None,
        ignore=None,
        retries=5,
    ):
        in_dir = os.path.expanduser(in_dir)
        if out_dir is None:
            out_dir = in_dir
        else:
            out_dir = os.path.expanduser(out_dir)

        if ignore is None:
            ignore = _DEFAULT_IGNORE
        elif not ignore:
            ignore = None

        logger.info(in_dir)
        relevant_files_list = self.relevant_files(in_dir, prompt, ignore=ignore)
        logger.info("Considering")
        logger.info(relevant_files_list)

        target_files_list = self.target_files(
            in_dir,
            prompt,
            system="\n".join(
                [
                    _BASE_PROMPT,
                    f"You've decided that these are the files you needed to consider: {relevant_files_list}",
                ]
            ),
            ignore=ignore,
        )
        logger.info("Changing")
        logger.info(target_files_list)
        return self.edit_files_considering(
            [os.path.join(in_dir, pth) for pth in relevant_files_list],
            [os.path.join(out_dir, pth) for pth in target_files_list],
            prompt,
            after_save=after_save,
            retries=retries,
        )


_BASE_PROMPT = "You are an extremely experienced and knowledgeable programmer. A genie in human form, able to bend source code to your will in ways your peers can only marvel at."

_DEFAULT_IGNORE = r"(^__pycache__|^node_modules|^env|^venv|^\..*|~$|\.pyc$|Thumbs\.db$|^build[\\/]|^dist[\\/]|^coverage[\\/]|\.log$|\.lock$|\.bak$|\.swp$|\.swo$|\.tmp$|\.temp$|\.class$|^target$|^Cargo\.lock$)"


def _edit_files_considering(
    m, relevant_files_list, target_files_list, prompt, after_save, retries
):
    relevant_files = {
        pth: util.slurp(pth) for pth in relevant_files_list if os.path.isfile(pth)
    }
    for pth in target_files_list:
        m.edit_file(
            pth,
            "\n".join(
                [
                    _BASE_PROMPT,
                    f"You've decided that these are the files you needed to consider: {relevant_files}",
                ]
            ),
            prompt,
            after_save=after_save,
        )
        logger.info(f"EDITED {pth}")
    return {
        "considered": relevant_files_list,
        "changed": target_files_list,
    }


def _relevant_files(
    m, in_dir, prompt, ignore=None, focus=None, must_exist=False, retries=5
):
    if ignore is None:
        ignore = _DEFAULT_IGNORE
    project_tree = util.tree(in_dir, ignore=ignore, focus=focus)
    files_list = m.generate_checked(
        util.mk_local_files(in_dir, must_exist=must_exist),
        "\n".join(
            [
                _BASE_PROMPT,
                f"The directory tree of the directory you've been asked to work on is {project_tree}. What files does the users' query require you to consider? Return a JSON-formatted list of relative pathname strings and no other content.",
            ]
        ),
        prompt,
        retries=retries,
    ).content
    return files_list


def _target_files(
    m, in_dir, prompt, system=None, ignore=None, focus=None, must_exist=False, retries=5
):
    if ignore is None:
        ignore = _DEFAULT_IGNORE
    project_tree = util.tree(in_dir, ignore=ignore, focus=focus)
    files_list = m.generate_checked(
        util.mk_local_files(in_dir, must_exist=must_exist),
        "\n".join(
            [
                system or _BASE_PROMPT,
                f"The directory tree of the directory you've been asked to work on is {project_tree}. What files does the users' query require you to edit? Return a JSON-formatted list of relative pathname strings and no other content.",
            ]
        ),
        prompt,
        retries=retries,
    ).content
    return files_list
