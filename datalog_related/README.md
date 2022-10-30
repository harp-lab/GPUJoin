## Souffle
### Installation on Ubuntu 22.04
```
sudo wget https://souffle-lang.github.io/ppa/souffle-key.public -O /usr/share/keyrings/souffle-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/souffle-archive-keyring.gpg] https://souffle-lang.github.io/ppa/ubuntu/ stable main" | sudo tee /etc/apt/sources.list.d/souffle.list
sudo apt update
sudo apt install souffle
```

### Running Souffle program
```
souffle --version
souffle -F . -D . tc.dl
souffle -F . -D . -o tc_dl tc.dl
./tc_dl
time ./tc_dl -j 4
time ./tc_dl -j 8
```


### References
- [Install Souffle](https://souffle-lang.github.io/install.html)
