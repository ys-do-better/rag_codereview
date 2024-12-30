import re
import nltk
import wordninja
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

# 指定nltk_data的路径
nltk.data.path.append("D:/桌面/原始数据/env/Scripts/nltk_data")

print(nltk.data.path)

# 处理C++代码文本（去除注释，去除@字符和数字，拆分长单词）
def process_code_text(code_text):
    # 去除C++代码中的注释（单行注释 // 和 多行注释 /*...*/）
    code_text = re.sub(r'//.*|/\*.*?\*/', '', code_text, flags=re.DOTALL)
    
    # 去除@字符
    code_text = re.sub(r'@', '', code_text)
    
    # 去除数字
    code_text = re.sub(r'\d+', '', code_text)

    #去除+、-和，字符
    code_text = re.sub(r'[+\-,]', '', code_text)  # 去除 "+"、"-" 和 ","

    # # 拆分长单词（如 HavingDefaultValue -> having default value）
    # code_text = ' '.join(wordninja.split(code_text))
    
    # # 保留除句号外的所有标点符号
    # code_text = re.sub(r'(?<!\.)[^\w\s]', '', code_text)  # 去除非句号的标点符号
    code_lower = code_text.lower()

    tokens = word_tokenize(code_lower)

    # 3. 对分词结果进行词形还原
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # 将词形还原后的标记重新合并成字符串
    processed_text = " ".join(lemmatized_tokens)
    
    return processed_text

# 处理评论文本（去除符号，扩展缩写，去除标点，限制长度）
def process_comment_text(comment_text):
    # 去除符号，如 "@" 等
    comment_text = re.sub(r'@[\w]+', '', comment_text)
    
    # 扩展缩写（例如，What's -> What is）
    abbreviations = {"What's": "What is", "I'm": "I am", "can't": "cannot", "doesn't": "does not"}
    for abbr, full_form in abbreviations.items():
        comment_text = comment_text.replace(abbr, full_form)
    
    # 去除标点符号，保留语义词
    comment_text = re.sub(r'[^\w\s]', '', comment_text)

    comment_lower = comment_text.lower()

    tokens = word_tokenize(comment_lower)

    # 3. 对分词结果进行词形还原
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # 将词形还原后的标记重新合并成字符串
    processed_comment = " ".join(lemmatized_tokens)
    
    return processed_comment

#读取excel文件获取数据
import pandas as pd
file_path = "D:/桌面/原始数据/filtered_data_simple_match.xlsx" # 替换为你的Excel文件路径
datas = pd.read_excel(file_path)
#获取excel文件中的‘diff_hunk’和‘body’列
data = datas[['diff_hunk', 'body']]
# 对每一行的代码和评论进行处理
for i in range(len(data)):
    data.loc[i, 'diff_hunk'] = process_code_text(data.loc[i, 'diff_hunk'])  # 修改此行
    data.loc[i, 'body'] = process_comment_text(data.loc[i, 'body'])  # 修改此行
#保存处理后的数据
data.to_excel('processed_data.xlsx', index=False)

print('数据处理完成！')
