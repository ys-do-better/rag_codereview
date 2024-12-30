import random
import nltk
from nltk.corpus import wordnet
import pandas as pd

# 下载nltk资源
nltk.download('punkt')
nltk.download('wordnet')

# 同义词替换函数
def synonym_replacement(sentence):
    words = nltk.word_tokenize(sentence)
    new_words = []
    for word in words:
        # 获取词语的同义词
        synonyms = wordnet.synsets(word)
        if synonyms:
            # 随机选择一个同义词
            synonym = random.choice(synonyms).lemmas()[0].name()
            new_words.append(synonym)
        else:
            new_words.append(word)
    return ' '.join(new_words)

# 生成增强版本
def augment_text(text):
    augmented_texts = set()  # 使用集合避免重复

    # 同义词替换
    augmented_texts.add(synonym_replacement(text))

    # 句式变化: 改变句子中部分词语的顺序或结构
    words = nltk.word_tokenize(text)
    if len(words) > 3:
        random.shuffle(words)  # 打乱词语顺序
        augmented_texts.add(' '.join(words))

    # 添加一些固定的句式变化
    if "seem" in text:
        augmented_texts.add(text.replace("seem", "appear"))
    if "appear" in text:
        augmented_texts.add(text.replace("appear", "seem"))

    return list(augmented_texts)[:3]  # 限制最多生成3个变种

# 读取Excel文件
file_path = "D:/桌面/原始数据/filtered_excel_file.xlsx"  # 替换为你的Excel文件路径
df = pd.read_excel(file_path)

# 初始化新的列表，用于存储增强后的数据
new_rows = []

# 对原始数据进行处理
for i, review in enumerate(df['body']):
    # 原始数据
    row = [review]  # 每行首先是原始数据
    
    # 生成增强数据并添加到新行列表
    augmented_reviews = augment_text(review)
    row.extend(augmented_reviews)  # 将增强数据添加到同一行
    
    # 将该行添加到 new_rows 中
    new_rows.append(row)

# 创建新的 DataFrame，并设置动态列名
columns = ['body'] + [f'augmented_{i+1}' for i in range(3)]  # 原始数据 + 最多3列增强数据
augmented_df = pd.DataFrame(new_rows, columns=columns)

# 保存数据到 Excel 文件
augmented_df.to_excel("augmented_data_with_multiple_columns.xlsx", index=False)
print("Augmented data with multiple columns has been saved to 'augmented_data_with_multiple_columns.xlsx'.")

