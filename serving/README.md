线上部署相关
线上部署主要有两种思路：

1. all-in-one
这种方案即将model 在server 端load 进来，然后直接预测后返回结果
   
2. tf-serving
tf-serving 具有热更新，支持多模型多版本，异步调用，高可用等特性，所以也推荐使用tf-serving。使用了tf-serving后，完整的路线变为：
   client --> backend --> rpc/rest --> tf-serving
   
tf-serving的安装推荐docker 通过镜像安装，手动安装比较麻烦，安装参考：[tf serving with docker](https://tensorflow.google.cn/tfx/serving/docker?hl=zh-cn)

```shell
# Download the TensorFlow Serving Docker image and repo
docker pull tensorflow/serving

git clone https://github.com/tensorflow/serving
# Location of demo models
TESTDATA="$(pwd)/serving/tensorflow_serving/servables/tensorflow/testdata"

# Start TensorFlow Serving container and open the REST API port
docker run -t --rm -p 8501:8501 \
    -v "$TESTDATA/saved_model_half_plus_two_cpu:/models/half_plus_two" \
    -e MODEL_NAME=half_plus_two \
    tensorflow/serving &

# Query the model using the predict API
curl -d '{"instances": [1.0, 2.0, 5.0]}' \
    -X POST http://localhost:8501/v1/models/half_plus_two:predict

# Returns => { "predictions": [2.5, 3.0, 4.5] }
```

不过推荐用配置文件启动，如果走rpc，还需要暴露8500 端口
```shell
docker run -t --rm -p 8501:8501 -v "D:/Docker/models:/models" tensorflow/serving --model_config_file=/models/models.config
```

```editorconfig
model_config_list {
  config {
    name: 'model_name'
    base_path: '/path/to/model'
    model_platform: 'tensorflow'
  }
}
```

Tips: model文件下需要有一层版本，即models/model_name/01/saved_model.pb
