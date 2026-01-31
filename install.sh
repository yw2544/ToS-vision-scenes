sudo install -d /etc/apt/keyrings
curl -fsSL https://hub.unity3d.com/linux/keys/public | sudo gpg --dearmor -o /etc/apt/keyrings/unityhub.gpg
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/unityhub.gpg] https://hub.unity3d.com/linux/repos/deb stable main" | sudo tee /etc/apt/sources.list.d/unityhub.list
sudo apt update
sudo apt install -y unityhub

# xvfb is required if no server
sudo apt install -y xvfb

# set Editor install path
sudo mkdir -p ${HOME}/Unity/Hub/Editor
sudo chown -R "$USER":"$USER" ${HOME}/Unity/Hub/Editor
xvfb-run --auto-servernum unityhub --headless install-path --set ${HOME}/Unity/Hub/Editor

xvfb-run --auto-servernum unityhub --no-sandbox --headless install --version 2020.3.48f1 # --changeset b805b124c6b7

# Required packages
sudo apt install -y libgconf-2-4
sudo apt install -y assimp-utils

# From ubuntu-toolchain ppa
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt install -y gcc-9 libstdc++6