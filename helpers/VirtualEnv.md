## Virtual Environment 

### Initial Setup

```
python -m venv virEnv
```

Creates a new virtual environment named *env*.

### Activate Environment

```
virEnv\Scripts\activate.bat
```

### Freeze File

Create a file detailing the recipe of installed packages in a virtual environment.

```
pip freeze > requirements.txt
```

### Read Freeze File

```
pip install -r requirements.txt
```