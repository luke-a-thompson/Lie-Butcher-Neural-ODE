## Installation

We recommend using [Astral UV](https://docs.astral.sh/uv/) to install this package.

### Install Tutorial

#### 1. Install Python

Make sure you have Python 3.10 or newer installed. You can download it from [python.org](https://www.python.org/downloads/). Alternatively, [you can use UV itself to manage your Python install](https://docs.astral.sh/uv/guides/install-python/).

#### 2. Set Up a Virtual Environment (Recommended)

It's best to use a virtual environment to keep dependencies clean:

```bash
python3 -m venv .venv
source .venv/bin/activate    # On Windows use: .venv\Scripts\activate
```

#### 3. Install with `uv` (Recommended)

If you don’t have `uv` installed, you can install it with:

```bash
pip install uv
```

Then install all dependencies:

```bash
uv sync
```

if you're using VSCode or Cursor, you may need to select the Python3 interpreter available in `./venv`

For more details, see the [Astral UV documentation](https://docs.astral.sh/uv/).
