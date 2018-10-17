# Installs the pythond executable at ~/bin/pythond

mkdir -p ~/bin
cp scripts/gpujob_dispatching/pythond.py ~/bin
echo "python ~/bin/pythond.py \"\$@\"" >> ~/bin/pythond
chmod +x ~/bin/pythond