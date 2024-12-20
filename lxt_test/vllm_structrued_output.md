
# vLLM 结构化输出完整教程

## 1. response_format 参数

这是最基本的结构化输出方式，支持 `json_object` 类型。

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

# 基本 JSON 输出
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-v0.1",
    messages=[
        {"role": "user", "content": "生成一个用户信息"}
    ],
    response_format={"type": "json_object"}
)

# 输出示例
# {
#     "name": "张三",
#     "age": 25,
#     "email": "zhangsan@example.com"
# }
```

## 2. guided_json 参数

使用 JSON Schema 来严格控制输出格式。

```python
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-v0.1",
    messages=[
        {"role": "user", "content": "生成一个用户信息"}
    ],
    guided_json={
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
            },
            "interests": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 5
            }
        },
        "required": ["name", "age"]
    }
)
```

## 3. guided_regex 参数

使用正则表达式控制输出格式。

```python
# 日期格式
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-v0.1",
    messages=[
        {"role": "user", "content": "生成一个标准日期"}
    ],
    guided_regex=r"\d{4}-\d{2}-\d{2}"
)

# 手机号格式
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-v0.1",
    messages=[
        {"role": "user", "content": "生成一个中国手机号"}
    ],
    guided_regex=r"1[3-9]\d{9}"
)

# 邮箱格式
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-v0.1",
    messages=[
        {"role": "user", "content": "生成一个邮箱地址"}
    ],
    guided_regex=r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
)
```

## 4. guided_choice 参数

限定输出必须是预设选项之一。

```python
# 简单选择
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-v0.1",
    messages=[
        {"role": "user", "content": "对这个提议表态"}
    ],
    guided_choice=["同意", "不同意", "需要更多信息"]
)

# 多选项场景
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-v0.1",
    messages=[
        {"role": "user", "content": "为这篇文章选择一个分类"}
    ],
    guided_choice=[
        "技术",
        "设计",
        "管理",
        "营销",
        "运营"
    ]
)

# 带描述的选项
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-v0.1",
    messages=[
        {"role": "user", "content": "评估这个项目的优先级"}
    ],
    guided_choice={
        "P0": "最高优先级，必须立即处理",
        "P1": "高优先级，本周必须完成",
        "P2": "中等优先级，两周内完成",
        "P3": "低优先级，可以延后处理"
    }
)
```

## 5. guided_grammar 参数

使用自定义语法规则控制输出格式。

```python
# 数学表达式语法
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-v0.1",
    messages=[
        {"role": "user", "content": "生成一个简单的数学表达式"}
    ],
    guided_grammar="""
    expression ::= number operator number
    operator ::= "+" | "-" | "*" | "/"
    number ::= [0-9]+
    """
)

# SQL 查询语法
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-v0.1",
    messages=[
        {"role": "user", "content": "生成一个查询用户表的SQL"}
    ],
    guided_grammar="""
    query ::= "SELECT" columns "FROM" table_name where_clause?
    columns ::= "*" | column_list
    column_list ::= column ("," column)*
    column ::= [a-zA-Z_][a-zA-Z0-9_]*
    table_name ::= [a-zA-Z_][a-zA-Z0-9_]*
    where_clause ::= "WHERE" condition
    condition ::= column operator value
    operator ::= "=" | ">" | "<" | ">=" | "<=" | "!="
    value ::= number | string
    number ::= [0-9]+
    string ::= "'" [^']* "'"
    """
)
```

## 6. 组合使用示例

### 6.1 JSON 输出配合正则验证

```python
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-v0.1",
    messages=[
        {"role": "user", "content": "生成用户联系信息"}
    ],
    guided_json={
        "type": "object",
        "properties": {
            "phone": {
                "type": "string",
                "pattern": "^1[3-9]\\d{9}$"
            },
            "email": {
                "type": "string",
                "pattern": "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
            }
        },
        "required": ["phone", "email"]
    }
)
```

### 6.2 选项配合 JSON 结构

```python
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-v0.1",
    messages=[
        {"role": "user", "content": "生成一个任务状态报告"}
    ],
    guided_json={
        "type": "object",
        "properties": {
            "task_id": {"type": "string"},
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "completed", "failed"]
            },
            "priority": {
                "type": "string",
                "enum": ["P0", "P1", "P2", "P3"]
            }
        },
        "required": ["task_id", "status", "priority"]
    }
)
```

## 7. 错误处理最佳实践

```python
from openai import OpenAIError

def generate_structured_output(prompt, structure_type, structure_params):
    try:
        response = client.chat.completions.create(
            model="mistralai/Mistral-7B-v0.1",
            messages=[
                {"role": "user", "content": prompt}
            ],
            **{structure_type: structure_params}
        )
        return response.choices[0].message.content
    except OpenAIError as e:
        print(f"API 错误: {str(e)}")
        return None
    except Exception as e:
        print(f"其他错误: {str(e)}")
        return None

# 使用示例
result = generate_structured_output(
    "生成用户信息",
    "guided_json",
    {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
    }
)
```

## 8. 注意事项

1. **参数互斥**
   - `response_format`、`guided_json`、`guided_regex`、`guided_choice` 和 `guided_grammar` 这些参数通常是互斥的
   - 一次请求中只能使用其中一个参数

2. **模型支持**
   - 不同模型对这些参数的支持程度可能不同
   - 建议查看具体模型的文档确认支持的功能

3. **性能考虑**
   - 复杂的约束可能会影响生成速度
   - 建议根据实际需求选择最简单的约束方式

4. **验证输出**
   - 即使使用了约束，也建议对输出进行额外验证
   - 特别是在处理关键数据时

这个教程涵盖了 vLLM 中所有结构化输出的参数用法。您可以根据具体需求选择合适的参数来控制输出格式。