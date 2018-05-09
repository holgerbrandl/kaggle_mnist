


## Data Prep

```bash
kaggle competitions download -c digit-recognizer

```



## Processing


```bash

talsync(){ 
srcDir=$(pwd | sed 's/\/Users\/brandl//g')
# continue here
}

rsync --delete -avx --exclude target --exclude project ~/projects/deep_learning/mnist_kotlin_example/ brandl@talisker:~/projects/deep_learning/mnist_kotlin_example

```

```bash
screen -R ox

cd ~/projects/deep_learning/kaggle_yelp_rest_pics

# http://www.gubatron.com/blog/2017/07/20/how-to-run-your-kotlin-gradle-built-app-from-the-command-line/

gradle run 2>&1 | tee yelp.$(date +'%Y%m%d').log
mailme "yelp done in $(pwd)"

```


## Cuda support

https://askubuntu.com/questions/917356/how-to-verify-cuda-installation-in-16-04

```bash
nvidia-smi
nvcc --version
```


## Misc



Kaggle submission whth

```bash
kaggle competitions submit -c digit-recognizer -f submission.csv -m "Message"
```