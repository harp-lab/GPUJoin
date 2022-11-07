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
g++ tc_dl.cpp -I .
```

### Error
```shell
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ ./tc_dl
./tc_dl: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `GLIBCXX_3.4.29' not found (required by ./tc_dl)
./tc_dl: /usr/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.32' not found (required by ./tc_dl)
./tc_dl: /usr/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.33' not found (required by ./tc_dl)
./tc_dl: /usr/lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.34' not found (required by ./tc_dl)
arsho@thetagpu06:/lus/theta-fs0/projects/dist_relational_alg/shovon/GPUJoin/datalog_related$ g++ -o tc_2 tc_dl.cpp 
tc_dl.cpp:2:10: fatal error: souffle/CompiledSouffle.h: No such file or directory
    2 | #include "souffle/CompiledSouffle.h"
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~
compilation terminated.
```


### References
- [Install Souffle](https://souffle-lang.github.io/install.html)
