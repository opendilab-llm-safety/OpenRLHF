import ray
ray.init(address="ray://10.140.0.137:10001")  # 使用Ray原生协议
# 或
# ray.init(address="http://10.140.0.137:8266")  # 使用HTTP协议

@ray.remote
def hello():
    return "hello"

future = hello.remote()
print(ray.get(future))
