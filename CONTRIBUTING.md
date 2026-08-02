# Contributing to PhotoChoose

Thank you for helping improve PhotoChoose.

For a substantial feature or behavior change, open an issue first so the approach can be discussed.
Small fixes and documentation improvements can go directly to a pull request.

## Development setup

PhotoChoose uses Python 3.11.

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
~~~

On Windows, create the environment with `py -3.11 -m venv .venv` and activate it with
`.venv\Scripts\activate`.

Before opening a pull request, run:

~~~bash
python -m unittest discover -s tests
python -m ruff check src tests
~~~

## Pull requests

- Branch from the latest `main`.
- Keep each pull request focused on one change.
- Add or update tests for behavior changes.
- Update user documentation when controls or workflows change.
- Describe the user-visible result and the operating systems you tested.
- Do not commit personal photographs, PhotoChoose catalog data, credentials, or generated builds.
- Do not add or replace bundled models without documenting their source, license, and checksum.

Pull requests are squash-merged, so a clear pull request title becomes the final commit title.

By contributing, you agree that your contribution is licensed under the repository's
GPL-3.0-only license and that you have the right to submit it.
