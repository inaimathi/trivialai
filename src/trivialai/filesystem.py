import os

from . import util


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

    def edit_directory(
        self,
        in_dir,
        prompt,
        after_save=None,
        out_dir=None,
        ignore_regex=None,
        retries=5,
    ):
        base = "You are an extremely experienced and knowledgeable programmer. A genie in human form, able to bend source code to your will in ways your peers can only marvel at."
        in_dir = os.path.expanduser(in_dir)
        if out_dir is None:
            out_dir = in_dir
        else:
            out_dir = os.path.expanduser(out_dir)

        if ignore_regex is None:
            ignore_regex = r"(^__pycache__|^node_modules|^env|^venv|^\..*|~$|\.pyc$|Thumbs\.db$|^build[\\/]|^dist[\\/]|^coverage[\\/]|\.log$|\.lock$|\.bak$|\.swp$|\.swo$|\.tmp$|\.temp$|\.class$|^target$|^Cargo\.lock$)"
        elif not ignore_regex:
            ignore_regex is None

        print(in_dir)
        project_tree = util.tree(in_dir, ignore_regex)
        files_list = self.generate_checked(
            util.mk_local_files(in_dir, must_exist=False),
            "\n".join(
                [
                    base,
                    f"The directory tree of the directory you've been asked to work on is {project_tree}. What files does the users' query require you to consider? Return a JSON-formatted list of relative pathname strings and no other content.",
                ]
            ),
            prompt,
        ).content
        print(f"   Considering {files_list}")
        files = {
            fl: util.slurp(os.path.join(in_dir, fl))
            for fl in files_list
            if os.path.isfile(os.path.join(in_dir, fl))
        }

        change_files_list = self.generate_checked(
            util.mk_local_files(in_dir, must_exist=False),
            "\n".join(
                [
                    base,
                    f"The project tree of the project you've been asked to work on is {project_tree}.",
                    f"You've decided that these are the files you needed to consider: {files}",
                    "What files does the users' query require you to make changes to? Return a JSON-formatted list of relative pathnames of type [RelativePath] and no other commentary or content",
                ]
            ),
            prompt,
        ).content

        print(f"   Changing {change_files_list}")
        for pth in change_files_list:
            self.edit_file(
                os.path.join(out_dir, pth),
                "\n".join(
                    [
                        base,
                        f"The project tree of the project you've been asked to work on is {project_tree}.",
                        f"You've decided that these are the files you needed to consider: {files}",
                    ]
                ),
                prompt,
                after_save=after_save,
            )
