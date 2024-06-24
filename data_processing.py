import json
import pandas as pd
import random

# 读取 JSON 文件
with open('./data/gossipcop_v3-4_story_based_fake.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 创建一个空的 DataFrame
df = pd.DataFrame(columns=['id', 'content', 'label', 'type'])

# 解析 JSON 数据并填充 DataFrame
for key, value in data.items():
    id_ = value['origin_id']
    content = value['generated_text']
    label = 1 if value['origin_label'] == 'fake' else 0
    df = df.append({'id': id_, 'content': content, 'label': label}, ignore_index=True)

# 随机打乱数据
df = df.sample(frac=1).reset_index(drop=True)

# 分割数据集，85% 为 training，15% 为 test
train_size = int(0.85 * len(df))
df.loc[:train_size, 'type'] = 'training'
df.loc[train_size:, 'type'] = 'test'

# 保存为 CSV 文件
df.to_csv('./data/StoryBasedFake4/sample.csv', index=False)
