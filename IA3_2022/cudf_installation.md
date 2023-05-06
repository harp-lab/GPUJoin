- Update `.zshrc`:
```
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/arsho/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/arsho/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/home/arsho/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/arsho/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```
- Create and activate new `conda` environment:
```
conda create --name gpu_env
conda activate gpu_env
```
- Install packages:
```
conda install -c rapidsai -c nvidia -c numba -c conda-forge \
    cudf=22.06 python=3.9 cudatoolkit=11.2
conda install -c rapidsai -c nvidia -c numba -c conda-forge \
    cudf=22.06 cugraph python=3.9 cudatoolkit=11.2    
```
- Create a test script `example.py`:
```
import pandas as pd
import cudf

s = cudf.Series([1,2,3,None,4])
print(s)
```
- Run the program:
```
python example.py
```
### References
- [cudf installation docs](https://github.com/rapidsai/cudf)
- [nvidia rapids kit cheatsheet](https://images.nvidia.com/aem-dam/Solutions/ai-data-science/rapids-kit/accelerated-data-science-print-getting-started-cheat-sheets.pdf)
- [blog article on conda usage](https://carpentries-incubator.github.io/introduction-to-conda-for-data-scientists/02-working-with-environments/index.html)
- [cugraph installation docs](https://github.com/rapidsai/cugraph#conda)
