# Installs the pythony executable at ~/bin/pythony

mkdir -p ~/bin
cp scripts/command_logger/autologcmd.py ~/bin/
echo "python ~/bin/autologcmd.py python \"\$@\"" >> ~/bin/pythony
chmod +x ~/bin/pythony
