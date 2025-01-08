import os

from . import util

_BASE_PROMPT = "You are an extremely experienced and knowledgeable programmer. A genie in human form, able to bend source code to your will in ways your peers can only marvel at."

_DEFAULT_IGNORE = r"(^__pycache__|^node_modules|^env|^venv|^\..*|~$|\.pyc$|Thumbs\.db$|^build[\\/]|^dist[\\/]|^coverage[\\/]|\.log$|\.lock$|\.bak$|\.swp$|\.swo$|\.tmp$|\.temp$|\.class$|^target$|^Cargo\.lock$)"


def relevant_files(m, in_dir, prompt, ignore=None, focus=None, must_exist=False):
    if ignore is None:
        ignore = _DEFAULT_IGNORE
    project_tree = util.tree(in_dir, ignore=ignore, focus=focus)
    files_list = m.generate_checked(
        util.mk_local_files(in_dir, must_exist=must_exist),
        "\n".join(
            [
                _BASE_PROMPT,
                f"The directory tree of the directory you've been asked to work on is {project_tree}. What files does the users' query require you to consider or change? Return a JSON-formatted list of relative pathname strings and no other content.",
            ]
        ),
        prompt,
    ).content
    return files_list


class FilesystemMixin:
    def edit_file(
        self, file_path, system, prompt, after_save=None, consider_current=True
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

    def relevant_files(self, in_dir, prompt, ignore=None, focus=None, must_exist=None):
        return relevant_files(
            self, in_dir, prompt, ignore=ignore, focus=focus, must_exist=must_exist
        )

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

        print(in_dir)
        files_list = self.relevant_files(in_dir, prompt, ignore=ignore)
        files = {
            fl: util.slurp(os.path.join(in_dir, fl))
            for fl in files_list
            if os.path.isfile(os.path.join(in_dir, fl))
        }

        joined = "\n    - ".join(files_list)
        print(f"   Changing \n{joined}")
        for pth in files_list:
            print(f"    x {pth}")
            self.edit_file(
                os.path.join(out_dir, pth),
                "\n".join(
                    [
                        _BASE_PROMPT,
                        f"You've decided that these are the files you needed to consider: {files}",
                    ]
                ),
                prompt,
                after_save=after_save,
            )
