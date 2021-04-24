# Nothing
Originally from [bagel-tensorflow](https://github.com/AlumiK/bagel-tensorflow).
> Bagel
>
> Litmus

# How to use?
> configs/global_config.yaml
> 
> configs/hyper_params.yaml
> 
> configs/mad_threshold.yaml

## 1. global_config.yaml
### 1. DATA_ROOT: the root path for dataset(cases)
- test
    - case1
    - case2
    - ...
    - exclude (files belong to here wille not be loaded)
- data

### 2. fault_injection
- start: the time when the configuration change starts
- end: the time when the configuration change ends

## 2. hyper_params.yaml
> just some paramters...
>
> Want to know more? Just find the usage in python scripts, you will know it.

## 3. mad_threshold.yaml
> This yaml file is used for mad algorithm. 
> 
> There are various types of metrics, also we provide a `default` threshold!