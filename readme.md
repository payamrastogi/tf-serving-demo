```shell
./venv/bin/saved_model_cli show --dir ./model/1 --tag_set serve --signature_def serving_default
```

```shell  
docker run -p 8501:8501 --mount type=bind,source=/home/<user>/workspace/tf-serving-demo/model,target=/models/model -e MODEL_NAME=model -t tensorflow/serving

```

```shell
 python rest_client.py
```

#### References
- https://keras.io/examples/keras_recipes/tf_serving/
- https://github.com/tensorflow/docs/blob/master/site/en/r1/guide/saved_model.md#cli-to-inspect-and-execute-savedmodel
- https://www.tensorflow.org/tfx/serving/docker