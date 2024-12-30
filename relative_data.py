import pandas as pd
import re

# 读取Excel文件
file_path = "D:/桌面/原始数据/valid_length_filtered_output.xlsx" # 替换为你的Excel文件路径
df = pd.read_excel(file_path)

# 过滤函数：检查评论内容和代码修订中是否有相同的单词
def is_relevant_comment(body, diff_hunk):
    """
    判断评论内容是否与代码修订内容相关（只要有相同的单词或字符就算相关）
    """
    # 将评论和代码修订转换为小写
    body_lower = body.lower()
    diff_hunk_lower = diff_hunk.lower()
    
    # 提取评论和代码修订中的所有单词
    body_words = set(re.findall(r'\b\w+\b', body_lower))  # 使用正则提取单词
    diff_hunk_words = set(re.findall(r'\b\w+\b', diff_hunk_lower))  # 提取代码中的单词
    
    # 判断评论和代码中是否有交集（即有相同的单词）
    return not body_words.isdisjoint(diff_hunk_words)  # 如果有交集，返回True

# 过滤噪声评论的函数
def filter_noise_comments(df):
    """
    过滤掉与代码修订无关的噪声评论
    """
    filtered_comments = []
    
    for index, row in df.iterrows():
        body = row['body']  # 获取评论内容
        diff_hunk = row['diff_hunk']  # 获取代码修订内容
        
        # 判断评论内容与代码修订内容是否相关
        if is_relevant_comment(body, diff_hunk):
            filtered_comments.append(row)
    
    # 将过滤后的结果转换回 DataFrame
    return pd.DataFrame(filtered_comments)

# 调用过滤函数
filtered_df = filter_noise_comments(df)

# 查看过滤后的数据
print(filtered_df)

# 保存处理后的结果
filtered_df.to_excel("filtered_data_simple_match.xlsx", index=False)
