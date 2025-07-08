# Setting up the Python environment
* To setup the Python programming environment, you need to do following steps:
  1. Install pyenv
  2. Install Python
* Windows and Mac users go different route for step1, while step 2 is common.

# Install
## Installing pyenv
### Windows
1. Install WSL2 (Windows service for linux ver.2); following is an example in Windows 10
  1. Open "Windows Power Shell" or "Terminal" as Administrator.
  2. Type `wsl --set-default-version 2` (for safe)
  3. Type `wsl --install -d Ubuntu`.
  4. After installation is done, open `Ubuntu` in Application.

* When above doesn't work, check "Windows の機能の有効化、または無効化", then "Linux 用 Windows サブシステム", "仮想マシンプラットフォーム" is ON.
* Note that WSL makes home directory (`/home/your_name`) which is different from the Windows user directory (`C:\Users\your_name`).
* You can access Windows system from Ubuntu like: `cd` to Desktop by `cd /mnt/c/Users/your_name/Desktop/`.
* It is useful to make symbolic link between Ubuntu and Windows. So, in *Ubuntu terminal (not PowerShell)*, do
  ```bash
  cd
  ln -s /mnt/c/Users/your_user_name/Desktop/ desktop
  ```

* If some commands are unavailable (e.g. vi), install them by `sudo apt install vim`.

2. Install dependencies
  ```bash
  sudo apt update
  sudo apt -y upgrade
  ```
then
  ```bash
  sudo apt -y install build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev curl \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
  ```

3. Install pyenv
  ```bash
  curl https://pyenv.run | bash
  ```

### Mac
1. Install Homebrew (if not installed)
  + copy command in https://brew.sh/ja/ in the terminal.
2. Install pyenv via Homebrew
  ```bash
  brew install pyenv
  ```

### Linux
* `git clone https://github.com/pyenv/pyenv.git ~/.pyenv`

## pyenv configuration
1. Update Bash Configuration. Execute following three lines.
  ```bash
  echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
  echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
  echo 'eval "$(pyenv init -)"' >> ~/.bashrc
  ```
* if you are using zsh, replace `~/.bashrc` with `~/.zshrc`.

2. Reload Bash Configuration
  ```bash
  source ~/.bashrc
  ```
* If you are using zsh, replace `~/.bashrc` with `~/.zshrc`.

##  Install Python using pyenv
1. List available Python versions
  ```bash
  pyenv install --list
  ```

2. Choose a version and install it (e.g., Python 3.9.5)
  ```bash
  pyenv install 3.9.5
  ```
* It will take some time ...

3. Set Global Python Version
  ```bash
  pyenv global 3.9.5
  ```

4. Verify Installation
  ```bash
  python --version
  ```

## Installing pip
* `pip` is the Python package manager.
* If pyenv is installed, pip is automatically installed.
* To not installed, do the following commands to install (both Windows WSL2 and MacOS).
  1. `curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py`
  2. `python get-pip.py`
* After installing pip, you can download the library like: `pip install numpy`.

## Python install from source
* When you don't have permission or OS is old, installing Python via pyenv does'nt work.
* In that case, you need to insall Python from source code.
1. Make some directory and go there: `mkdir python`, `cd python`.
2. Get the tar.xz file from web, using `wget`.
3. `tar jxvf Python.x.x.x.tar.xz`
4. `./configure --prefix=/your_home/python`.
5. `make` then `make install`.
* `pip` is also install. Do not forget to set PATH, and alias to map `python` -> `python3` and `pip` -> `pip3`.

### Troubleshooting
* `'ImportError: No module named '_tkinter', please install the python3-tk package'` (Windows)
  + Do `sudo apt install python3-tk`.

# VScode
* Visual Studio code (VS code) is a free code editor made by Microsoft.
* Works both on Windows and Mac.
## Setup
1. Download the installer from https://code.visualstudio.com/
2. Install extensions(拡張機能)
  + "Japanese Language Pack" (if you want).
  + "Python"

## Wrirting Python script
1. Open a folder where Python script exists
2. After writing the code, save file
3. The "run" command (">" button) to run the script

* You can change the Python interpreter (like pyenv) by clicking the "Python ..." in the bottom bar.

# Google Colaboratory
* Google Colab is a free, cloud-based platform where you can write and run Python code using Jupyter notebooks
* You can use expensive GPUs (graphical proceccing unit) with Google Cloab for free (with some limitations)

### Setup
1. Sign in with your Google Account (if exists).
2. Go to Google Colaboratory page in your web browser.
    * https://colab.research.google.com/notebooks/intro.ipynb
3. Create a New Notebook
    * "File(ファイル)" --> "New Notebook(ノートブックを新規作成)" to start a new Python notebook.
4. Copy your notebook to Google Drive.
    * Keep your notebook to Google Drive by clicking "Copy to Drive(ドライブにコピー)".
4. Use the Notebook
    * "+Code(コード)" will insert the code cell in your notebook.
    * In a code cell, you can input Python code.
    * With ">" button, the code will be execured.
5. Saving and Sharing
    * "File" --> "Save" will save your notebook to Google Drive.
    * Click the folder button in left panel then folders are open.
    * You file is save to "drive" --> "Mydrive" (by default).

### Notebook-specific grammar
* Putting `!` indicates the *shell command*.
* Putting `%` indicates the *magic command*.
    * e.g. To change the directory, specify `%cd` (not `!cd` because it generates new shell).
* You can directly edit the Python script (.py) file in Google Colab; click the .py file from the directory tree. 

### Using GPU
1. "Runtime" tab.
2. "Change the type of Runtime".
3. Choose GPU you want.
