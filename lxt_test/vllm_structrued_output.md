# vLLM 结构化输出完整教程

## 1. Chat Completion 接口的结构化输出

在 chat.completions.create() 接口中，支持以下 guided 参数来控制输出格式，需要通过 `extra_body` 字典传入。

### 1.1 guided_json 参数

使用 JSON Schema 来严格控制输出格式。

```python
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-v0.1",
    messages=[
        {"role": "user", "content": "生成一个用户信息"}
    ],
    extra_body={
        "guided_json": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "用户姓名"
                },
                "age": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 120,
                    "description": "用户年龄"
                },
                "contact": {
                    "type": "object",
                    "properties": {
                        "email": {
                            "type": "string",
                            "format": "email"
                        },
                        "phone": {
                            "type": "string",
                            "pattern": "^\\d{11}$"
                        }
                    },
                    "required": ["email"]
                }
            },
            "required": ["name", "age"]
        }
    }
)
```

### 1.2 guided_regex 参数

使用正则表达式控制输出格式。

```python
# 日期格式
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-v0.1",
    messages=[
        {"role": "user", "content": "生成一个标准日期"}
    ],
    extra_body={
        "guided_regex": r"\d{4}-\d{2}-\d{2}"
    }
)
```

### 1.3 guided_choice 参数

限定输出必须是预设选项之一。

```python
# 简单选择
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-v0.1",
    messages=[
        {"role": "user", "content": "对这个提议表态"}
    ],
    extra_body={
        "guided_choice": ["同意", "不同意", "需要更多信息"]
    }
)
```

### 1.4 guided_grammar 参数

使用自定义语法规则控制输出格式。

```python
# 数学表达式语法
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-v0.1",
    messages=[
        {"role": "user", "content": "生成一个简单的数学表达式"}
    ],
    extra_body={
        "guided_grammar": """
        expression ::= number operator number
        operator ::= "+" | "-" | "*" | "/"
        number ::= [0-9]+
        """
    }
)
```

## 2. Completion 接口的结构化输出

在 completions.create() 接口中，支持使用 response_format 和其他 guided 参数来控制 JSON 输出格式。

```python
response = client.completions.create(
    model="mistralai/Mistral-7B-v0.1",
    prompt="生成一个用户信息",
    extra_body={
        "response_format": {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                }
            }
        }
    }
)
```

## 3. 错误处理最佳实践

```python
from openai import OpenAIError

def generate_structured_output(prompt, structure_type, structure_params):
    try:
        response = client.chat.completions.create(
            model="mistralai/Mistral-7B-v0.1",
            messages=[
                {"role": "user", "content": prompt}
            ],
            extra_body={
                structure_type: structure_params
            }
        )
        return response.choices[0].message.content
    except OpenAIError as e:
        print(f"API 错误: {str(e)}")
        return None
    except Exception as e:
        print(f"其他错误: {str(e)}")
        return None
```

## 4. 注意事项

1. **接口区分**
   - chat.completions 接口支持 guided_json、guided_regex、guided_choice 和 guided_grammar
   - completions 接口支持 response_format 和其他 guided参数

2. **参数位置**
   - vLLM 的额外参数需要放在 `extra_body` 中
   - `extra_headers` 目前仅支持 `X-Request-Id`

3. **参数互斥**
   - 结构化输出参数（guided_json、guided_regex、guided_choice、guided_grammar）是互斥的
   - 一次请求中只能使用其中一个参数

4. **模型支持**
   - 不同模型对这些参数的支持程度可能不同
   - 建议查看具体模型的文档确认支持的功能

5. **验证输出**
   - 即使使用了约束，也建议对输出进行额外验证
   - 特别是在处理关键数据时

这个教程区分了 chat.completions 和 completions 两个接口的结构化输出参数用法。您可以根据使用的接口和具体需求选择合适的参数来控制输出格式。