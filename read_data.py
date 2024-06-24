# 忽略所有警告信息，通常在生产环境中不推荐这样做，但在开发和调试过程中有时会用到。
import warnings

warnings.filterwarnings("ignore")

# 导入必要的库
import pandas as pd
import numpy as np
# 从自定义的features模块中导入manual_features、segmentation_text和clean_regex函数
from features.building_features import manual_features, segmentation_text, clean_regex
# 从自定义的data模块中导入split函数
from data.utils import split

# 设置随机种子，确保结果的可重复性
np.random.seed(0)


# 定义一个函数，用于准备输入数据
def prepare_input(dataset='MultiSourceFake', segments_number=10, n_jobs=-1, emo_rep='frequency', return_features=True,
                  text_segments=False, clean_text=True):
    # 加载数据集，数据集的路径是相对于当前脚本的相对路径
    content = pd.read_csv('./data/{}/sample.csv'.format(dataset))
    content_features = []

    """提取特征，分割文本，清理文本。"""
    # 如果需要返回特征，则调用manual_features函数提取文本情感特征
    if return_features:
        content_features = manual_features(n_jobs=n_jobs, path='./features', model_name=dataset, segments_number=segments_number, emo_rep=emo_rep).transform(content['content'])

    # 如果需要文本分段，则调用segmentation_text函数进行分段
    """在分段时，我们已经对文本进行了清理，只保留 DOTS (.) 。"""
    if text_segments:
        content['content'] = segmentation_text(segments_number=segments_number).transform(content['content'])
    # 如果需要清洗文本，则使用clean_regex函数进行清洗
    elif clean_text:
        content['content'] = content['content'].map(lambda text: clean_regex(text, keep_dot=True))

    # 调用split函数将数据集分为训练集、验证集和测试集
    train, dev, test = split(content, content_features, return_features)
    # 返回分割后的数据集
    return train, dev, test


if __name__ == '__main__':
    # 调用prepare_input函数，准备输入数据
    train, dev, test = prepare_input(dataset='MultiSourceFake', segments_number=10, n_jobs=-1, text_segments=True)
