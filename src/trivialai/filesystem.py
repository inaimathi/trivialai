import os

from . import llm, util


class CodeMonkeyMixin:
    def edit_file(self, file_path, system, prompt, after_save=None):
        cont = self.generate(
            f"{system}. What changes would you make to the file {file_path}? Return only the new contents of {file_path} and no other information.",
            prompt,
        ).content
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

        def _local_files(resp):
            try:
                loaded = util.loadch(resp)
                if type(loaded) is not list:
                    raise llm.TransformError(
                        "relative-file-response-not-list", raw=resp
                    )
                return [util.relative_path(in_dir, f) for f in loaded]
            except Exception:
                pass
            raise llm.TransformError("relative-file-translation-failed", raw=resp)

        print(in_dir)
        project_tree = util.tree(in_dir, ignore_regex)
        files_list = self.generate_checked(
            _local_files,
            "\n".join(
                [
                    base,
                    f"The project tree of the project you've been asked to work on is {project_tree}. What files does the users' query require you to consider? Return a JSON-formatted list of relative pathname strings and no other content.",
                ]
            ),
            prompt,
        ).content
        print(f"   Considering {files_list}")
        files = {fl: util.slurp(os.path.join(in_dir, fl)) for fl in files_list}

        change_files_list = self.generate_checked(
            _local_files,
            "\n".join(
                [
                    base,
                    f"The project tree of the project you've been asked to work on is {project_tree}.",
                    f"You've decided that these are the files you needed to consider: {files}",
                    "What files does the users' query require you to make changes to? Return a JSON-formatted list of relative pathnames and no other content",
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


def developer(client, in_dir, prompt, after_save=None, out_dir=None, ignore_regex=None):
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

    def _local_files(resp):
        try:
            loaded = util.loadch(resp)
            if type(loaded) is not list:
                raise llm.TransformError("relative-file-response-not-list", raw=resp)
            return [util.relative_path(in_dir, f) for f in loaded]
        except Exception:
            pass
        raise llm.TransformError("relative-file-translation-failed", raw=resp)

    print(in_dir)
    project_tree = util.tree(in_dir, ignore_regex)
    files_list = client.generate_checked(
        _local_files,
        "\n".join(
            [
                base,
                f"The project tree of the project you've been asked to work on is {project_tree}. What files does the users' query require you to consider? Return a JSON-formatted list of relative pathname strings and no other content.",
            ]
        ),
        prompt,
    ).content
    print(f"   Considering {files_list}")
    files = {fl: util.slurp(os.path.join(in_dir, fl)) for fl in files_list}
    if files_list:
        for f in files_list:
            fl = os.path.join(in_dir, f)
            if not os.path.isfile(fl):
                continue
            with open(fl, "r") as in_file:
                files[f] = in_file.read()

    change_files_list = client.generate_checked(
        _local_files,
        "\n".join(
            [
                base,
                f"The project tree of the project you've been asked to work on is {project_tree}.",
                f"You've decided that these are the files you needed to consider: {files}",
                "What files does the users' query require you to make changes to? Return a JSON-formatted list of relative pathnames and no other content",
            ]
        ),
        prompt,
    ).content
    print(f"   Changing {change_files_list}")
    for pth in change_files_list:
        cont = client.generate(
            "\n".join(
                [
                    base,
                    f"The project tree of the project you've been asked to work on is {project_tree}.",
                    f"You've decided that these are the files you needed to consider: {files}",
                    f"What changes would you make to the file {pth}? Return only the new contents of {pth} and no other information.",
                ]
            ),
            prompt,
        ).content
        full_file_path = os.path.join(out_dir, pth)
        util.spit(full_file_path, util.strip_md_code(cont))
        if after_save is not None:
            after_save(full_file_path)


# filesystem.developer(oclient, "~/projects/mycroft-server", """Walk the python files and replace any HTTP error we send that looks like `""status": "nope"` with `"status": "error", "message": "<insert helpful message here>" """)
