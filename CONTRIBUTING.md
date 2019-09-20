# How to contribute?

All contributions are welcome !

We just set a few conditions:

* Respect the [PEP8](https://www.python.org/dev/peps/pep-0008/). Our favorite
tool in that way is `flake8` (configuration in `.flake8` file). You can run
`flake8` on your own way. Be careful, *With great power, comes great
responsibility.* :) It is also available here as a pre-commit hook. To activate
the hook, just run the following command after the project installation:

```
(venv)$ pre-commit install -f
```

* Pass the unit tests before pushing up any branch. Here the recommanded tool
  is `pytest`. Let's keep it full-green! To automatize this process, you can
  add the command in the `.git/hooks/pre-push` file, so as to run it each time
  you invokes `git push`.

As a remark, if you find any bug in the code, feel free to push new unit tests.
