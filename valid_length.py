import pandas as pd
import re

# 读取Excel文件
file_path = "D:\桌面\原始数据\cm_slava77_PR_comment_info.xlsx"  # 替换为你的Excel文件路径
df = pd.read_excel(file_path)

# 判断文本是否符合长度要求
def is_valid_length(text):
    word_count = len(text.split())
    return 3 <= word_count <= 128

# # 去除无意义的评论（例如“fixed”）
# def is_meaningful_comment(comment):
#     return not bool(re.match(r'^\s*(fixed|done|resolved|approved)\s*$', comment, re.IGNORECASE))

# 判断代码函数是否过大（假设代码大于某个阈值时被认为是"气味"）
def is_valid_code(code, max_code_length=500):
    # 假设代码过长的判定是基于字符长度，可以根据实际情况调整
    return len(code) <= max_code_length

# 过滤掉不符合要求的行
df_filtered = df[
    df['body'].apply(is_valid_length) & 
    # df['body'].apply(is_meaningful_comment) & 
    df['diff_hunk'].apply(is_valid_code)
]

# 保存处理后的数据
df_filtered.to_excel("valid_length_filtered_output.xlsx", index=False)

print("Filtered data has been saved to 'filtered_output.xlsx'.")
